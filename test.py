#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile
import time
import logging
from typing import List, Tuple

import madmom

import subprocess
import json

def get_video_duration(video_path):      # 用 ffprobe 获取视频时长      
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"获取视频时长失败: {result.stderr}")
    return float(result.stdout.strip())

# ==================== 1. 配置日志信息 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 2. 视频切割器类定义 ====================
class DanceVideoCutter:
    
    
    """舞蹈视频自动切割器，基于音乐节拍将视频按8拍一组进行分割"""

    def __init__(self, ffmpeg_path: str = 'ffmpeg'):
        """
        初始化切割器
        :param ffmpeg_path: ffmpeg可执行文件的路径，默认为系统环境变量中的'ffmpeg'
        """
        self.ffmpeg_path = ffmpeg_path
        # 检查ffmpeg是否可用
        try:
            subprocess.run([self.ffmpeg_path, '-version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("FFmpeg not found. Please ensure it's installed and in your PATH.")
            raise RuntimeError("FFmpeg not found")

    def extract_audio(self, video_path: str) -> str:
        """(1) 视频分离：从视频中提取音频轨并保存为WAV文件"""
        logger.info(f"Extracting audio from video: {video_path}")
        # 在系统的临时目录下创建一个临时文件，使用完毕后会自动删除
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()

        # ffmpeg命令：-i 输入文件，-vn 忽略视频流，-acodec pcm_s16le 音频编码为16bit PCM，
        # -ar 44100 采样率44.1kHz，-ac 1 单声道，-y 覆盖输出文件
        cmd = [self.ffmpeg_path, '-i', video_path, '-vn',
               '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
               '-y', temp_audio.name]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio extracted to temporary file: {temp_audio.name}")
            return temp_audio.name
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr.decode()}")
            raise

    def detect_beats(self, audio_path: str) -> List[float]:
        """(2)(3) 音频标准化与节拍检测：
        输入音频路径，返回节拍的时间戳列表（单位：秒）"""
        logger.info(f"Detecting beats from audio: {audio_path}")
        try:
            # 创建 Madmom 的处理器链
            # RNNBeatProcessor: 使用预训练的循环神经网络模型分析音频，生成节拍激活函数（概率信号）
            # 参考：https://madmom.readthedocs.io/en/latest/modules/features/beats.html#madmom.features.beats.RNNBeatProcessor
            beat_processor = madmom.features.beats.RNNBeatProcessor()
            # BeatTrackingProcessor: 通过动态贝叶斯网络（DBN）对概率信号进行解码，得到精确的节拍时间戳
            # fps=100: Madmom内部以100Hz的帧率进行处理，这与RNN的输出采样率相匹配
            # 参考：https://madmom.readthedocs.io/en/latest/modules/features/beats.html#madmom.features.beats.BeatTrackingProcessor
            tracker = madmom.features.beats.BeatTrackingProcessor(fps=100)

            # 执行节拍检测
            beats = tracker(beat_processor(audio_path))

            # beats是一个numpy数组，直接转换为列表返回
            logger.info(f"Detected {len(beats)} beats. First beat at {beats[0]:.2f}s, last at {beats[-1]:.2f}s")
            
                        # 打印每一个节拍的具体时间点
            logger.info("=== 检测到的所有节拍时间点 ===")
            for idx, beat_time in enumerate(beats.tolist(), start=1):
                logger.info(f"第 {idx:2d} 拍: {beat_time:.3f} 秒")
            logger.info("================================\n")
            
            return beats.tolist()
        except Exception as e:
            logger.error(f"Beat detection failed: {e}")
            raise


    def group_into_eight_beats(self,video_path: str, beats: List[float]) -> List[Tuple[float, float]]:
        """(4) 八拍分组与切割区间计算：
        输入节拍时间戳列表，按每8拍一组进行分组，返回各组的(起始时间, 结束时间)区间列表"""
        video_duration = get_video_duration(video_path) #获取视频总时长
        intervals = []
        total_beats = len(beats)
        if total_beats == 0:
            return intervals

        # # 以第一个有效节拍为起点
        start_beat_idx = 0
        # if len(beats) > 0 and beats[0] > 1: #CHANGE:强制从0秒为第一段起点
        #     beats.insert(0,0.0)
        # 每8拍为一组
        group_size = 8

        for i in range(start_beat_idx, total_beats, group_size):
            group_start_time = beats[i]
            # 确定组结束的节拍索引：如果剩余不足8拍，则取最后一个节拍
            # end_beat_idx = min(i + group_size - 1, total_beats - 1)
            # group_end_time = beats[end_beat_idx]

            # CHANGE：结束时间取「下一组的起始节拍时间」
            if i + group_size < total_beats:
            # 不是最后一组，结束时间取下一组的起始节拍
                group_end_time = beats[i + group_size]
            else:
            # 最后一组，结束时间取视频总时长
                group_end_time = video_duration

            # 核心：打印切割点
            logger.info(f"==================================================")
            logger.info(f"第 {len(intervals)+1} 组切割点确立：")
            logger.info(f"→ 开始时间：{group_start_time:.3f} 秒")
            logger.info(f"→ 结束时间：{group_end_time:.3f} 秒")
            logger.info(f"==================================================\n")
            
            intervals.append((group_start_time, group_end_time))

        logger.info(f"Grouped beats into {len(intervals)} segments based on 8 beats each.")
        return intervals

    # def cut_video_segment(self, video_path: str, start_time: float, end_time: float, output_path: str):
    #     """(5) 视频分段切割：根据给定的区间切割视频片段"""
    #     logger.info(f"Cutting segment from {start_time:.2f}s to {end_time:.2f}s -> {output_path}")

        # ## 构建FFmpeg命令，采用流复制模式（-c copy），提高切割效率，避免重新编码
        # # -ss: 起始时间（放在-i之前可以加快定位速度），-to: 结束时间，-c copy: 直接复制流
        # # -avoid_negative_ts make_zero: 处理时间戳，使输出文件的时间戳从0开始
        # # 参考：https://ffmpeg.org/ffmpeg.html#Main-options
        # cmd = [self.ffmpeg_path,
        #        '-i', video_path,
        #        '-ss', str(start_time),
        #        '-to', str(end_time),
        #        '-c', 'copy',
        #        '-avoid_negative_ts', 'make_zero',
        #        '-y',  # 覆盖已存在的输出文件
        #        output_path]

        # try:
        #     subprocess.run(cmd, check=True, capture_output=True)
        #     logger.info(f"Segment cut successfully: {output_path}")
        # except subprocess.CalledProcessError as e:
        #     logger.error(f"Segment cutting failed: {e.stderr.decode()}")
        #     raise
#CHNAGE：
    def cut_video_segment(self, video_path: str, start_time: float, end_time: float, audio_out: str, mute_out: str):
        logger.info(f"Cutting segment from {start_time:.2f}s to {end_time:.2f}s ")
       
        logger.info(f"切割带音乐视频: {audio_out}")
        duration = end_time - start_time

        # ###
        # 同一时间段切割两份视频
        # audio_out：带原声音乐视频
        # mute_out：无音乐静音视频
        # ###
        # 1. 切割【带音乐】版本，保留原音   
        # cmd_audio = [
        #     self.ffmpeg_path,
        #     '-ss', str(start_time),  #CHANGE：先i后ss改为 先ss后i
        #     '-i', video_path, 
        #     '-to', str(duration),
        #     '-c', 'copy',   #！！直接用原视频流，速度快，但剪出来的视频片段重复
        #     '-avoid_negative_ts', 'make_zero',
        #     '-y',
        #     audio_out
        # ]

        cmd_audio = [
            self.ffmpeg_path,
            '-y',
            '-hwaccel', 'qsv',          # 启用Intel核显硬解源视频
            '-ss', f"{start_time:.6f}",
            '-i', video_path,
            '-t', f"{duration:.6f}",
            '-c:v', 'h264_qsv', # Intel QSV硬件编码H.264，降低CPU占用 # 视频重新编码  ！！速度慢很多，但剪出来片段准确，衔接精准
            '-c:a', 'aac',    # 统一AAC音频，兼容全平台 # 音频重新编码
            '-pix_fmt', 'yuv420p',      # 标准像素格式，
            '-ar', '48000',
            '-avoid_negative_ts', '1',   #自动把负时间戳改成 0
            audio_out
        ]
        subprocess.run(cmd_audio, check=True, capture_output=True)

        # 2. 切割【静音无音乐】版本，去除音频
        logger.info(f"切割静音无音乐视频: {mute_out}")
        cmd_mute = [
            self.ffmpeg_path,
            "-y",
            '-hwaccel', 'qsv',          # 同样开启硬件解码
            "-ss", f"{start_time:.6f}",
            "-i", video_path,
            "-t", f"{duration:.6f}",
            "-c:v", "h264_qsv",    # 硬件编码
            "-pix_fmt", "yuv420p",
            "-an",                # 删除音频（无声）
            "-avoid_negative_ts", "1",  
            "-sn",
            mute_out       
        ]
        subprocess.run(cmd_mute, check=True, capture_output=True)

    def process_video(self, video_path: str, output_dir: str, num_retries: int = 3):
        """处理主流程：整合所有步骤，对单个视频进行切割"""
        

        # 获取视频文件名（不含扩展名）用于生成输出文件名
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        #CHANFE:# 确保输出目录存在
        # os.makedirs(output_dir, exist_ok=True)
        # 分别创建两个存放文件夹
        audio_dir = os.path.join(output_dir, f"{base_name}_带音乐")
        mute_dir = os.path.join(output_dir, f"{base_name}_无音乐")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(mute_dir, exist_ok=True)
        # 临时音频文件变量
        temp_audio = None

        try:
            # 1. 提取音频
            for attempt in range(num_retries):
                try:
                    temp_audio = self.extract_audio(video_path)
                    break
                except Exception as e:
                    logger.warning(f"Audio extraction attempt {attempt+1}/{num_retries} failed: {e}")
                    if attempt == num_retries - 1:
                        raise
                    time.sleep(1)

            # 2. 检测节拍
            beats = self.detect_beats(temp_audio)

            # 3. 八拍分组
            intervals = self.group_into_eight_beats(video_path,beats)

            if not intervals:
                logger.warning("No beat intervals generated. Video may be too short or no beats detected.")
                return

            # 4. 切割视频
            for idx, (start, end) in enumerate(intervals):

                # 打印真实要切割的时间点 
                logger.info(f"【FFMPEG 实际切割】片段 {idx+1}:")
                logger.info(f"  代码节拍切割点:  {start:.4f} → {end:.4f}")
                # logger.info(f"  真实执行命令: ffmpeg -ss {start:.4f} -i 视频 -to {end:.4f} 输出")
               
                # # 生成输出文件名：原视频名_8拍_序号.mp4
                #CHANGE:
                # output_filename = f"{base_name}_8beats_{idx+1:03d}.mp4"
                # output_path = os.path.join(output_dir, output_filename)
                # audio_file = os.path.join(output_dir, f"{base_name}_8beats_{idx+1:03d}_带音乐.mp4")
                # mute_file = os.path.join(output_dir, f"{base_name}_8beats_{idx+1:03d}_无音乐.mp4")
                
                audio_file = os.path.join(audio_dir, f"{base_name}_8beats_{idx+1:03d}.mp4")
                mute_file = os.path.join(mute_dir, f"{base_name}_8beats_{idx+1:03d}.mp4")
                #CHANGE:# # 如果文件已存在，且大小不为0，则跳过切割（假设已存在是完整的）
                # if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                #     logger.info(f"Output file {output_path} already exists, skipping cut.")
                #     continue
                # if os.path.exists(audio_file) and os.path.exists(mute_file):
                #     logger.info(f"两个版本都已存在，跳过: {idx+1}")
                #     continue

                for attempt in range(num_retries):
                    try:
                        #CHANGE: # self.cut_video_segment(video_path, start, end, output_path)
                        self.cut_video_segment(video_path, start, end, audio_file, mute_file)
                        break  # 切割成功，跳出重试循环
                    except Exception as e:
                        logger.warning(f"Segment {idx+1} cut attempt {attempt+1}/{num_retries} failed: {e}")
                        if attempt == num_retries - 1:
                            raise
                        time.sleep(1)

            logger.info(f"Successfully processed video {video_path}. Segments saved to {output_dir}")

        finally:
            # 清理临时音频文件，避免占用过多本地存储
            if temp_audio and os.path.exists(temp_audio):
                os.unlink(temp_audio)
                logger.info(f"Cleaned up temporary audio file: {temp_audio}")


# ==================== 3. 主程序入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dance Video Cutter based on Music Beats")
    parser.add_argument("input_video", help="Path to the input MP4 video file")
    parser.add_argument("-o", "--output", default="./output_segments", help="Output directory for cut segments")
    parser.add_argument("-r", "--retries", type=int, default=3, help="Number of retry attempts for failed operations")

    args = parser.parse_args()

    # 创建切割器实例并运行
    cutter = DanceVideoCutter()
    cutter.process_video(args.input_video, args.output, args.retries)

    