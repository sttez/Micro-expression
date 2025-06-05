import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import time
import cv2
from collections import deque
import traceback

from cnn_model import CNNFeatureExtractor
from lstm_model import MicroExpressionLSTM

# 检测器路径
FACE_CASCADE_PATH = r'data/haarcascade_frontalface_default.xml'
# LANDMARK_MODEL_PATH = r'utils/shape_predictor_68_face_landmarks.dat' # Not used in this function yet

# 情绪颜色映射（BGR格式）
EMOTION_COLORS = {
    'disgust': (0, 0, 255),       # 红色
    'happiness': (0, 255, 0),     # 绿色
    'others': (128, 128, 128),    # 灰色
    'repression': (128, 0, 128),  # 紫色
    'surprise': (255, 255, 0)     # 青色
}

# 图像大小（模型输入要求）
IMAGE_SIZE = 128

class MicroExpressionPredictor:
    """微表情预测器"""

    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: 训练好的模型路径
            device: 运行设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 32

        # 标签映射
        self.label_map = {
            0: 'disgust',
            1: 'happiness',
            2: 'others',
            3: 'repression',
            4: 'surprise'
        }

        # 图像变换 (仅包含 ToTensor 和 Normalize)
        self.transform = transforms.Compose([
            transforms.ToTensor(), # PIL Image [0, 255] -> Tensor [0.0, 1.0]
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize grayscale [0.0, 1.0] -> [-1.0, 1.0]
        ])

        # 加载模型
        self._load_model(model_path)

    def _load_model(self, model_path):
        """加载训练好的模型"""
        print(f"Loading model from {model_path}")

        # 创建CNN特征提取器
        cnn_extractor = CNNFeatureExtractor()

        # 创建LSTM模型
        self.model = MicroExpressionLSTM(
            cnn_feature_extractor=cnn_extractor,
            input_size=32768,
            hidden_size1=128,
            hidden_size2=64,
            num_classes=5,
            sequence_length=32,
            dropout_rate=0.3
        )

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # For models saved without 'model_state_dict' key (e.g., older saves)
            # This might require careful handling depending on the exact save format.
            # Attempting direct load assumes the state_dict is the top-level object.
            try:
                 self.model.load_state_dict(checkpoint)
            except RuntimeError as e:
                 print(f"Error loading state_dict directly. Check checkpoint format: {e}")
                 # If direct load fails, you might need to inspect the checkpoint structure
                 # or handle specific keys if it's a partial save.
                 # For now, re-raise or handle appropriately.
                 raise e

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def predict_sequence(self, image_sequence_path):
        """预测图像序列的微表情

        Args:
            image_sequence_path: 包含图像序列的文件夹路径

        Returns:
            prediction: 预测的表情类别
            confidence: 置信度分数
            probabilities: 各类别的概率
        """
        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(image_sequence_path)
                              if f.endswith(('.jpg', '.png', '.jpeg'))])

        if len(image_files) < self.sequence_length:
            raise ValueError(f"序列长度不足，需要至少{self.sequence_length}帧")

        # 随机选择连续的32帧
        start_idx = np.random.randint(0, len(image_files) - self.sequence_length + 1)
        selected_files = image_files[start_idx:start_idx + self.sequence_length]

        # 加载图像序列
        images = []
        for img_file in selected_files:
            img_path = os.path.join(image_sequence_path, img_file)
            img = Image.open(img_path).convert('L')
            img_tensor = self.transform(img)
            images.append(img_tensor)

        # 堆叠成序列张量
        sequence_tensor = torch.stack(images).unsqueeze(0)  # (1, 32, 1, 128, 128)

        # 预测
        start_time = time.time()
        with torch.no_grad():
            sequence_tensor = sequence_tensor.to(self.device)
            outputs = self.model(sequence_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        inference_time = time.time() - start_time

        # 获取结果
        predicted_class = predicted.item()
        predicted_emotion = self.label_map[predicted_class]
        confidence_score = confidence.item()
        all_probabilities = probabilities.cpu().numpy()[0]

        print(f"\n预测结果:")
        print(f"  表情类别: {predicted_emotion}")
        print(f"  置信度: {confidence_score:.4f}")
        print(f"  推理时间: {inference_time:.3f}秒")
        print(f"\n各类别概率:")
        for i, prob in enumerate(all_probabilities):
            print(f"  {self.label_map[i]}: {prob:.4f}")

        return predicted_emotion, confidence_score, all_probabilities

    def predict_single_image(self, image_path):
        """预测单张图像（通过复制成序列）

        Args:
            image_path: 单张图像路径

        Returns:
            prediction: 预测的表情类别
            confidence: 置信度分数
        """
        # 加载图像
        img = Image.open(image_path).convert('L')
        img_tensor = self.transform(img)

        # 复制成序列
        sequence_tensor = img_tensor.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # (1, 32, 1, 128, 128)

        # 预测
        with torch.no_grad():
            sequence_tensor = sequence_tensor.to(self.device)
            outputs = self.model(sequence_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        predicted_emotion = self.label_map[predicted_class]
        confidence_score = confidence.item()

        print(f"\n单图预测结果:")
        print(f"  表情类别: {predicted_emotion}")
        print(f"  置信度: {confidence_score:.4f}")
        print(f"  注意: 单图预测可能不如序列预测准确")

        return predicted_emotion, confidence_score

    def batch_predict(self, sequence_paths):
        """批量预测多个序列

        Args:
            sequence_paths: 序列文件夹路径列表

        Returns:
            results: 预测结果列表
        """
        results = []

        for i, seq_path in enumerate(sequence_paths):
            print(f"\n处理序列 {i + 1}/{len(sequence_paths)}: {seq_path}")
            try:
                emotion, confidence, probs = self.predict_sequence(seq_path)
                results.append({
                    'path': seq_path,
                    'emotion': emotion,
                    'confidence': confidence,
                    'probabilities': probs
                })
            except Exception as e:
                print(f"  错误: {str(e)}")
                results.append({
                    'path': seq_path,
                    'error': str(e)
                })

        return results


# 实时视频流预测器（扩展功能）
class RealTimeMicroExpressionPredictor(MicroExpressionPredictor):
    """实时视频流微表情预测器"""

    def __init__(self, model_path, buffer_size=32, device='cuda'):
        super().__init__(model_path, device)
        self.buffer_size = buffer_size
        # 使用deque以便高效移除旧帧
        self.frame_buffer = deque(maxlen=buffer_size)
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if self.face_cascade.empty():
            raise RuntimeError(f"Error loading cascade classifier at {FACE_CASCADE_PATH}")

    def add_processed_frame(self, processed_img_tensor):
        """将预处理好的图像张量添加到缓冲区"""
        self.frame_buffer.append(processed_img_tensor)

    def predict_from_buffer(self):
        """对缓冲区中的帧序列进行预测"""
        if len(self.frame_buffer) < self.buffer_size:
            return None

        try:
            # 使用缓冲区中的所有帧 (已经是buffer_size长度)
            # frame_buffer contains tensors of shape (C, H, W), stack to (seq_len, C, H, W)
            sequence = torch.stack(list(self.frame_buffer))
            # Add batch dimension: (1, seq_len, C, H, W)
            sequence = sequence.unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(sequence)
                probabilities = nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class = predicted.item()
            predicted_emotion = self.label_map[predicted_class]
            confidence_score = confidence.item()

            return {
                'emotion': predicted_emotion,
                'confidence': confidence_score,
                'probabilities': probabilities.cpu().numpy()[0]
            }

        except Exception as e:
            print(f"预测失败: {e}")
            traceback.print_exc()
            return None

    def process_frame(self, frame):
        """处理单帧图像并返回预测结果

        Args:
            frame: BGR格式的OpenCV图像

        Returns:
            tuple: (处理后的帧, 概率列表) 或 None（如果处理失败）
        """
        try:
            # 转换为灰度图用于人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # 使用第一个检测到的人脸
                (x, y, w, h) = faces[0]
                
                # 裁剪人脸区域
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # 缩放和灰度化
                    face_resized = cv2.resize(face_roi, (IMAGE_SIZE, IMAGE_SIZE))
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    
                    # 转换为PIL Image，应用Transforms
                    pil_face = Image.fromarray(face_gray)
                    processed_tensor = self.transform(pil_face)
                    
                    # 添加到缓冲区
                    self.add_processed_frame(processed_tensor)
                    
                    # 如果缓冲区已满，进行预测
                    if len(self.frame_buffer) == self.buffer_size:
                        result = self.predict_from_buffer()
                        if result:
                            # 在原始帧上绘制结果
                            emotion = result['emotion']
                            confidence = result['confidence']
                            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                            
                            # 绘制人脸框
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            
                            # 绘制文本
                            text = f"{emotion}: {confidence:.2f}"
                            cv2.putText(frame, text, (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            
                            return frame, result['probabilities']
            
            return frame, [0.0] * 5  # 返回原始帧和零概率列表
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            return None


def run_camera_prediction(model_path, camera_index=0, predict_interval=10):
    """运行摄像头实时微表情预测

    Args:
        model_path: 训练好的模型路径
        camera_index: 摄像头索引
        predict_interval: 每隔多少新帧添加到缓冲区后，使用最新的32帧进行预测
    """
    predictor = RealTimeMicroExpressionPredictor(model_path, buffer_size=32) # Buffer needs to be SEQUENCE_LENGTH
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    print("Press 'q' to exit...")

    latest_prediction_result = None
    frames_processed_for_buffer = 0
    # Keep track of the last successfully processed face image tensor
    last_processed_face_tensor = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # 摄像头画面镜像翻转
        frame = cv2.flip(frame, 1)

        # 转换为灰度图用于人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = cv2.CascadeClassifier(FACE_CASCADE_PATH).detectMultiScale(gray, 1.3, 5)

        current_processed_face_tensor = None

        if len(faces) > 0:
            # 简化处理：只使用第一个检测到的人脸
            (x, y, w, h) = faces[0]

            # 裁剪人脸区域
            face_roi = frame[y:y+h, x:x+w]

            if face_roi.size > 0:
                try:
                    # 缩放和灰度化
                    face_resized = cv2.resize(face_roi, (IMAGE_SIZE, IMAGE_SIZE))
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

                    # 转换为PIL Image，应用Transforms (ToTensor, Normalize)
                    pil_face = Image.fromarray(face_gray)
                    processed_tensor = predictor.transform(pil_face)
                    
                    current_processed_face_tensor = processed_tensor
                    last_processed_face_tensor = processed_tensor # Update last successful tensor

                except Exception as e:
                    print(f"处理人脸区域失败: {e}")
                    traceback.print_exc()
                    # Fallback to placeholder if processing fails
                    if last_processed_face_tensor is not None:
                         current_processed_face_tensor = last_processed_face_tensor
                    else:
                         # Add a zero tensor if no previous successful frame exists
                         current_processed_face_tensor = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
        else:
            # 如果没有人脸，添加一个占位符图像张量 (例如全零图像)
            # 或者重复上一帧成功处理的人脸，这里用全零
            if last_processed_face_tensor is not None:
                 # Use the last successfully processed face if no face is detected in current frame
                 current_processed_face_tensor = last_processed_face_tensor
            else:
                 # If no face ever detected, add a zero tensor
                 current_processed_face_tensor = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)

        # 添加处理后的帧到缓冲区
        predictor.add_processed_frame(current_processed_face_tensor)
        frames_processed_for_buffer += 1

        # 只有当缓冲区满 SEQUENCE_LENGTH 且达到预测间隔时才进行预测
        if len(predictor.frame_buffer) == predictor.buffer_size and frames_processed_for_buffer >= predict_interval:
            result = predictor.predict_from_buffer()
            if result:
                latest_prediction_result = result
            frames_processed_for_buffer = 0 # 重置计数器

        # 人脸检测和绘制 (使用原始帧进行绘制，因为裁剪后的图像只用于预测)
        display_frame = frame.copy()
        
        # 绘制人脸框和预测结果
        if latest_prediction_result:
            emotion = latest_prediction_result['emotion']
            confidence = latest_prediction_result['confidence']
            text = f"{emotion}: {confidence:.2f}"
            # 根据情绪获取颜色，默认为白色
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            text_color = (255, 255, 255) # 白色字体

            # 再次检测人脸用于在当前帧上绘制，确保框选位置准确
            # (这里可以直接用上面检测到的faces，但为了清晰，再次进行检测)
            # faces_to_draw = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)

            for (x, y, w, h) in faces:
                # 绘制人脸框 (使用情绪颜色)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                # 绘制文本背景框
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                # 确保文本框在画面内
                text_bg_x1 = x
                text_bg_y1 = y - text_h - baseline
                text_bg_x2 = x + text_w
                text_bg_y2 = y
                
                # 边界检查和调整
                if text_bg_y1 < 0:
                    text_bg_y1 = y
                    text_bg_y2 = y + text_h + baseline
                    text_y_draw = y + text_h + baseline # Draw text below if no space above
                else:
                    text_y_draw = y - baseline # Draw text above

                cv2.rectangle(display_frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1) # Filled rectangle

                # 绘制文本
                cv2.putText(display_frame, text, (x, text_y_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

        # 显示摄像头画面
        cv2.imshow('Camera Feed - Micro Expression Prediction', display_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    # 请替换为您的模型路径
    #model_path = 'output/lstm_models/final_model.pth'
    model_path = 'experiments/microexpression_20250528_223307/models/final_model.pth'
    
    # 运行摄像头预测
    # predict_interval=10 表示每积累10帧新数据后，使用最新的32帧进行预测
    run_camera_prediction(model_path, predict_interval=10)

    # 以下是原有的序列和单图预测示例，可以注释或删除
    # # 创建预测器
    predictor = MicroExpressionPredictor(
        model_path=r'D:\zhh\桌面\zuizhong\experiments\microexpression_20250528_223307\models\final_model.pth',
        device='cuda'
    )

    # 预测序列示例
    sequence_path = 'data/CASME2-RAW-cropped/sub01/EP02_01f'
    if os.path.exists(sequence_path):
        emotion, confidence, probs = predictor.predict_sequence(sequence_path)

    # # 预测单图示例
    # single_image_path = 'data/CASME2-RAW-cropped/sub01/EP02_01f/img001.jpg'
    # if os.path.exists(single_image_path):
    #     emotion, confidence = predictor.predict_single_image(single_image_path)