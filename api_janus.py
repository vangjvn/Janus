import os
import asyncio
import random
import time
from datetime import datetime
import logging
import traceback
from functools import wraps
import gc
from typing import Optional

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn
from starlette import status
from starlette.responses import JSONResponse
from PIL import Image
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import VLChatProcessor

# 配置日志
# 修改日志配置，使其更详细
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('janus.log')
    ]
)
logger = logging.getLogger(__name__)


# 异常处理装饰器
def handle_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return wrapper


app = FastAPI()

# 配置静态文件目录
static_dir = "output_images"
os.makedirs(static_dir, exist_ok=True)
app.mount("/output_images", StaticFiles(directory=static_dir), name="output_images")

# 初始化模型
GPU_ID = 3
model_path = "/home/clover666/janus/models/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'

# 加载模型
vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path,
    language_config=language_config,
    trust_remote_code=True
)

if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda(GPU_ID)
else:
    vl_gpt = vl_gpt.to(torch.float16)

# 建议在模型加载后添加
vl_gpt.eval()  # 设置为评估模式

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu'

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


@torch.inference_mode()
def generate(input_ids, width, height, temperature=1, cfg_weight=5):
    parallel_size = 1
    torch.cuda.empty_cache()

    # 简化代码，保持与源代码一致的处理方式
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, 576), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(576):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=pkv
            )
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, width // 16, height // 16]
    )

    return generated_tokens, patches

def unpack(dec, width, height):
    """简化的 unpack 函数"""
    dec = dec.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    return dec[0]  # 只返回第一张图片


# 创建一个信号量来控制GPU访问
gpu_semaphore = asyncio.Semaphore(1)  # 限制为1，因为只有一个GPU


# 创建一个请求队列类
class RequestQueue:
    def __init__(self, max_queue_size: int = 10, timeout_seconds: int = 300):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.timeout_seconds = timeout_seconds
        self.processing = {}  # 记录正在处理的请求

    async def add_request(self, request_id: str, prompt: str) -> None:
        try:
            # 尝试将请求添加到队列，设置超时时间
            await asyncio.wait_for(
                self.queue.put((request_id, prompt, time.time())),
                timeout=5.0  # 5秒内如果队列满，就抛出异常
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="服务器队列已满，请稍后重试"
            )

    async def get_request(self) -> Optional[tuple]:
        if not self.queue.empty():
            request_id, prompt, start_time = await self.queue.get()
            # 检查是否超时
            if time.time() - start_time > self.timeout_seconds:
                return None
            self.processing[request_id] = start_time
            return request_id, prompt
        return None

    def remove_request(self, request_id: str) -> None:
        self.processing.pop(request_id, None)

    def get_queue_position(self, request_id: str) -> int:
        # 返回请求在队列中的位置
        position = 1  # 从1开始计数
        for _, _, _ in self.queue._queue:
            position += 1
        return position


# 初始化请求队列
request_queue = RequestQueue(max_queue_size=10, timeout_seconds=200)


# 修改API端点
@app.post("/api/generate")
@handle_exceptions
async def generate_image_api(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # 生成唯一请求ID
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(prompt)}"

        # 尝试获取GPU信号量
        try:
            # 非阻塞地检查GPU是否可用
            if gpu_semaphore.locked():
                # GPU正忙，将请求加入队列
                await request_queue.add_request(request_id, prompt)
                position = request_queue.get_queue_position(request_id)
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content={
                        "status": "queued",
                        "request_id": request_id,
                        "queue_position": position,
                        "message": f"请求已加入队列，当前位置：{position}"
                    }
                )

            async with gpu_semaphore:
                # 实际的图像生成代码
                start_time = time.time()
                seed = data.get("seed", 12345)
                guidance = data.get("guidance", 5)
                temperature = data.get("temperature", 1.0)

                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                if seed == 12345:
                    seed = random.randint(10000,100000000)
                np.random.seed(seed)

                width = height = 384

                with torch.no_grad():
                    messages = [
                        {'role': '<|User|>', 'content': prompt},
                        {'role': '<|Assistant|>', 'content': ''}
                    ]

                    text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                        conversations=messages,
                        sft_format=vl_chat_processor.sft_format,
                        system_prompt=''
                    )
                    text = text + vl_chat_processor.image_start_tag

                    input_ids = torch.LongTensor(tokenizer.encode(text)).to(cuda_device)
                    output, patches = generate(
                        input_ids,
                        width // 16 * 16,
                        height // 16 * 16,
                        temperature=temperature,
                        cfg_weight=guidance
                    )

                    image_array = patches[0].to(torch.float32).cpu().numpy().transpose(1, 2, 0)
                    image_array = np.clip((image_array + 1) / 2 * 255, 0, 255).astype('uint8')

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"image_{timestamp}.png"
                filepath = os.path.join(static_dir, filename)

                img = Image.fromarray(image_array).resize((768, 768), Image.LANCZOS)
                img.save(filepath)

                process_time = time.time() - start_time

                return {
                    "status": "success",
                    "code": 200,
                    "image_url": f"/output_images/{filename}",
                    "process_time": f"{process_time:.2f}秒"
                }

        finally:
            # 如果请求在队列中，移除它
            request_queue.remove_request(request_id)

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 添加查询请求状态的端点
@app.get("/api/status/{request_id}")
async def check_status(request_id: str):
    if request_id in request_queue.processing:
        return {
            "status": "processing",
            "message": "请求正在处理中"
        }
    position = request_queue.get_queue_position(request_id)
    if position > 0:
        return {
            "status": "queued",
            "queue_position": position,
            "message": f"请求在队列中，当前位置：{position}"
        }
    return {
        "status": "not_found",
        "message": "请求不存在或已完成"
    }


# 添加后台任务处理队列中的请求
async def process_queue():
    while True:
        try:
            if not gpu_semaphore.locked():
                request = await request_queue.get_request()
                if request:
                    request_id, prompt = request
                    # 处理请求...
                    # 这里需要实现异步处理逻辑
                    pass
        except Exception as e:
            logger.error(f"处理队列时发生错误: {str(e)}")
        await asyncio.sleep(1)  # 避免过于频繁的检查


# 在应用启动时启动队列处理任务
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())
# 添加启动日志
if __name__ == "__main__":
    logger.info("正在启动Janus API服务...")
    logger.info(f"使用设备: {cuda_device}")
    logger.info(f"静态文件目录: {static_dir}")
    uvicorn.run(app, host="0.0.0.0", port=7878, workers=1)