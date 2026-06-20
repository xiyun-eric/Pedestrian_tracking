"""
ReID 外观特征提取模块

用于提取目标的外观特征，支持多种轻量级模型：
  - OSNet: 轻量、快速，适合实时跟踪
  - ResNet50: 精度高，适合离线处理
  - MobileNet: 极轻量，适合嵌入式设备

特征提取后用于：
  1. 代价矩阵融合（IoU + 马氏距离 + 外观相似度）
  2. 遮挡恢复（基于外观匹配丢失目标）
  3. ID 切换减少（外观一致性约束）
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
from pathlib import Path
import torch


class ReIDExtractor:
    """
    ReID 外观特征提取器
    
    支持多种模型，默认使用 OSNet（轻量快速）
    """
    
    # 支持的模型配置
    MODEL_CONFIGS = {
        'osnet_x1_0': {
            'input_size': (256, 128),  # (H, W) - 行人标准尺寸
            'feature_dim': 512,
            'weights_url': 'https://github.com/Kaiyang-Zhou/deep-person-reid/releases/download/v1.0/osnet_x1_0_msmt17.pt'
        },
        'osnet_x0_75': {
            'input_size': (256, 128),
            'feature_dim': 512,
            'weights_url': 'https://github.com/Kaiyang-Zhou/deep-person-reid/releases/download/v1.0/osnet_x0_75_msmt17.pt'
        },
        'osnet_x0_5': {
            'input_size': (256, 128),
            'feature_dim': 512,
            'weights_url': 'https://github.com/Kaiyang-Zhou/deep-person-reid/releases/download/v1.0/osnet_x0_5_msmt17.pt'
        },
        'osnet_x0_25': {
            'input_size': (256, 128),
            'feature_dim': 512,
            'weights_url': 'https://github.com/Kaiyang-Zhou/deep-person-reid/releases/download/v1.0/osnet_x0_25_msmt17.pt'
        },
        'resnet50': {
            'input_size': (256, 128),
            'feature_dim': 2048,
            'weights_url': None
        },
        'mobilenet': {
            'input_size': (256, 128),
            'feature_dim': 1280,
            'weights_url': None
        },
    }
    
    def __init__(
        self,
        model_name: str = 'osnet_x1_0',
        device: str = 'cuda:0',
        weights_path: Optional[str] = None,
        use_torchreid: bool = True,
        smooth_alpha: float = 0.7,
        use_finetuned: bool = False,
    ):
        """
        初始化 ReID 特征提取器

        Args:
            model_name: 模型名称 (osnet_x1_0, osnet_x0_75, resnet50, mobilenet)
            device: 设备 (cuda:0, cpu)
            weights_path: 预训练权重路径
            use_torchreid: 是否使用 torchreid 库
            smooth_alpha: EMA平滑系数（新特征权重，低值=更稳定）
            use_finetuned: 是否优先使用 LoRA 微调后的权重
        """
        self.model_name = model_name
        self.device = device
        self.smooth_alpha = smooth_alpha
        self.use_finetuned = use_finetuned
        
        # 检测设备
        if device.startswith('cuda') and not torch.cuda.is_available():
            print('[ReID] CUDA 不可用，使用 CPU')
            self.device = 'cpu'
        
        config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS['osnet_x1_0'])
        self.input_size = config['input_size']
        self.feature_dim = config['feature_dim']
        
        self.model = None
        self.use_torchreid = use_torchreid
        
        # 尝试加载模型
        self._load_model(weights_path)
        
        # 特征缓存（用于平滑更新）
        self.feature_cache = {}
        
    def _load_model(self, weights_path: Optional[str] = None):
        """加载 ReID 模型"""
        try:
            if self.use_torchreid:
                self._load_torchreid_model(weights_path)
            else:
                self._load_simple_model(weights_path)
        except Exception as e:
            print(f'[ReID] 加载模型失败: {e}')
            print('[ReID] 使用简化特征提取（颜色直方图 + HOG）')
            self.model = None
            self.use_simple_features = True
    
    def _load_torchreid_model(self, weights_path: Optional[str] = None):
        """使用 torchreid 库加载模型"""
        try:
            import torchreid
            
            # 搜索权重文件的路径（按优先级）
            search_paths = []
            if weights_path:
                search_paths.append(Path(weights_path))

            # 微调权重（LoRA 合并后的完整模型）- 最高优先级
            project_root = Path(__file__).resolve().parents[2]
            if self.use_finetuned:
                search_paths.append(project_root / 'runs' / 'reid' / 'osnet_lora' / 'best.pth')
                search_paths.append(project_root / 'runs' / 'reid' / 'osnet_lora' / 'last.pth')

            # 项目目录 - 预训练权重（优先搜索项目根目录，然后是 weights 子目录）
            search_paths.append(project_root / f'{self.model_name}_msmt17.pt')
            search_paths.append(project_root / f'{self.model_name}_msmt17.pth')
            search_paths.append(project_root / f'{self.model_name}.pt')
            search_paths.append(project_root / f'{self.model_name}.pth')
            search_paths.append(project_root / 'weights' / f'{self.model_name}_msmt17.pt')
            search_paths.append(project_root / 'weights' / f'{self.model_name}_msmt17.pth')
            search_paths.append(project_root / 'weights' / f'{self.model_name}.pt')
            search_paths.append(project_root / 'weights' / f'{self.model_name}.pth')
            # 用户缓存目录
            search_paths.append(Path.home() / '.cache' / 'torchreid' / f'{self.model_name}_msmt17.pt')
            search_paths.append(Path.home() / '.cache' / 'torchreid' / f'{self.model_name}_msmt17.pth')
            
            # 查找权重文件
            weights_file = None
            for path in search_paths:
                if path.exists():
                    weights_file = path
                    break
            
            # 确定num_classes：如果有权重文件，先读取分类层大小
            num_classes = 1000
            if weights_file:
                state_dict = torch.load(str(weights_file), map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                # 从classifier.weight推断类别数
                if 'classifier.weight' in state_dict:
                    num_classes = state_dict['classifier.weight'].shape[0]
            
            # 创建模型（使用正确的类别数）
            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=num_classes,
                pretrained=False
            )
            
            # 加载预训练权重
            loaded = False
            if weights_file:
                state_dict = torch.load(str(weights_file), map_location=self.device)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.model.load_state_dict(state_dict, strict=True)

                # 判断权重类型
                is_finetuned = 'reid' in str(weights_file) or 'osnet_lora' in str(weights_file)
                tag = '[微调]' if is_finetuned else '[预训练]'
                print(f'[ReID] {tag} 已加载权重: {weights_file} (num_classes={num_classes})')
                loaded = True
            
            if not loaded:
                # 尝试下载默认权重
                config = self.MODEL_CONFIGS.get(self.model_name, {})
                if config.get('weights_url'):
                    print(f'[ReID] 尝试下载预训练权重...')
                    try:
                        save_dir = project_root / 'weights'
                        save_dir.mkdir(parents=True, exist_ok=True)
                        weights_path = torchreid.utils.download_model(
                            config['weights_url'],
                            save_dir=str(save_dir)
                        )
                        state_dict = torch.load(weights_path, map_location=self.device)
                        self.model.load_state_dict(state_dict, strict=False)
                        print(f'[ReID] 已下载并加载权重')
                        loaded = True
                    except:
                        print('[ReID] 下载失败，使用未预训练模型')
            
            if not loaded:
                print('[ReID] 未找到预训练权重，使用未预训练模型（特征质量较低）')
            
            # OSNet 在 eval 模式下 forward 自动跳过 classifier 层，
            # 直接返回 512 维特征向量，无需手动移除分类层
            # 参考: torchreid OSNet.forward() 中 if not self.training: return v
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.use_simple_features = False
            print(f'[ReID] 模型加载成功: {self.model_name}, 特征维度: {self.feature_dim}')
            
        except ImportError:
            print('[ReID] torchreid 未安装，使用简化特征')
            self.model = None
            self.use_simple_features = True
    
    def _load_simple_model(self, weights_path: Optional[str] = None):
        """加载简化模型（ResNet 特征）"""
        try:
            import torchvision.models as models
            
            if self.model_name.startswith('resnet'):
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                # 移除最后的全连接层
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            elif self.model_name.startswith('mobilenet'):
                self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            else:
                # 默认使用 ResNet
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.use_simple_features = False
            print(f'[ReID] 简化模型加载成功')
            
        except Exception as e:
            print(f'[ReID] 简化模型加载失败: {e}')
            self.model = None
            self.use_simple_features = True
    
    def extract_feature(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        track_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        提取单个目标的外观特征
        
        Args:
            image: 原始图像 (H, W, 3)
            bbox: 边界框 [x1, y1, x2, y2]
            track_id: 轨迹ID（用于特征平滑）
            use_cache: 是否使用特征缓存
        
        Returns:
            feature: 特征向量 (feature_dim,)
        """
        # 裁剪目标区域
        x1, y1, x2, y2 = map(int, bbox)
        
        # 确保边界框在图像范围内
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            # 返回零向量
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # 提取特征
        if self.use_simple_features or self.model is None:
            feature = self._extract_simple_feature(crop)
        else:
            feature = self._extract_deep_feature(crop)
        
        # 特征平滑（与历史特征融合）
        if use_cache and track_id is not None:
            if track_id in self.feature_cache:
                old_feature = self.feature_cache[track_id]
                # 指数移动平均（使用配置的smooth_alpha）
                feature = self.smooth_alpha * feature + (1 - self.smooth_alpha) * old_feature
            self.feature_cache[track_id] = feature.copy()
        
        # 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        
        return feature
    
    def _extract_deep_feature(self, crop: np.ndarray) -> np.ndarray:
        """使用深度模型提取特征"""
        # 预处理
        crop_resized = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
        
        # 转换为 RGB（如果需要）
        if crop_resized.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop_resized
        
        # 归一化
        crop_normalized = crop_rgb.astype(np.float32) / 255.0
        
        # 标准化（ImageNet 标准）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        crop_normalized = (crop_normalized - mean) / std
        
        # 转换为 tensor
        tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.float().to(self.device)  # 确保使用 float32 类型
        
        # 提取特征
        with torch.no_grad():
            feature = self.model(tensor)
        
        # 转换为 numpy
        feature = feature.squeeze().cpu().numpy()
        
        return feature
    
    def _extract_simple_feature(self, crop: np.ndarray) -> np.ndarray:
        """使用简化方法提取特征（颜色直方图 + HOG）"""
        features = []
        
        # 1. 颜色直方图（HSV）
        if crop.shape[2] == 3:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # H 通道直方图
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-8)
            features.extend(h_hist)
            
            # S 通道直方图
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-8)
            features.extend(s_hist)
            
            # V 通道直方图
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
            v_hist = v_hist.flatten() / (v_hist.sum() + 1e-8)
            features.extend(v_hist)
        
        # 2. 简化 HOG 特征
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.shape[2] == 3 else crop
        gray_resized = cv2.resize(gray, (64, 128))
        
        # 计算梯度
        gx = cv2.Sobel(gray_resized, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray_resized, cv2.CV_32F, 0, 1)
        
        # 梯度幅值和方向
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180
        
        # 分块统计（简化版）
        cell_size = 8
        num_bins = 9
        hog_features = []
        
        for i in range(0, 128, cell_size):
            for j in range(0, 64, cell_size):
                cell_mag = mag[i:i+cell_size, j:j+cell_size]
                cell_angle = angle[i:i+cell_size, j:j+cell_size]
                
                # 直方图
                hist = np.zeros(num_bins)
                for k in range(num_bins):
                    mask = (cell_angle >= k * 20) & (cell_angle < (k + 1) * 20)
                    hist[k] = cell_mag[mask].sum()
                
                hist = hist / (hist.sum() + 1e-8)
                hog_features.extend(hist)
        
        features.extend(hog_features)
        
        # 3. LBP 特征（局部二值模式）
        lbp = self._compute_lbp(gray_resized)
        lbp_hist = np.histogram(lbp, bins=256, range=(0, 256))[0]
        lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-8)
        features.extend(lbp_hist[:64])  # 只取前64个
        
        feature = np.array(features, dtype=np.float32)
        
        # 如果特征维度不够，补零
        if len(feature) < self.feature_dim:
            feature = np.pad(feature, (0, self.feature_dim - len(feature)))
        elif len(feature) > self.feature_dim:
            feature = feature[:self.feature_dim]
        
        return feature
    
    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """计算 LBP 特征"""
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        return lbp
    
    def extract_features_batch(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        track_ids: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            image: 原始图像
            bboxes: 边界框数组 (N, 4)
            track_ids: 轨迹ID列表
        
        Returns:
            features: 特征数组 (N, feature_dim)
        """
        features = []
        
        for i, bbox in enumerate(bboxes):
            track_id = track_ids[i] if track_ids else None
            feature = self.extract_feature(image, bbox, track_id)
            features.append(feature)
        
        return np.array(features, dtype=np.float32)
    
    def compute_similarity(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
    ) -> float:
        """
        计算两个特征的相似度（余弦相似度）
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
        
        Returns:
            similarity: 相似度值 (0-1)
        """
        # 归一化
        f1 = feature1 / (np.linalg.norm(feature1) + 1e-8)
        f2 = feature2 / (np.linalg.norm(feature2) + 1e-8)
        
        # 余弦相似度
        similarity = np.dot(f1, f2)
        
        # 转换到 0-1 范围
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def compute_distance(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
    ) -> float:
        """
        计算两个特征的距离
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
        
        Returns:
            distance: 距离值 (0-1)
        """
        similarity = self.compute_similarity(feature1, feature2)
        return 1 - similarity
    
    def clear_cache(self):
        """清除特征缓存"""
        self.feature_cache.clear()
    
    def remove_from_cache(self, track_id: int):
        """从缓存中移除指定轨迹的特征"""
        if track_id in self.feature_cache:
            del self.feature_cache[track_id]


class FeatureBuffer:
    """
    特征缓冲区
    
    用于存储和管理轨迹的历史特征，支持：
    - 特征历史记录
    - 特征平滑更新
    - 特征一致性检查
    """
    
    def __init__(
        self,
        max_history: int = 30,
        smooth_alpha: float = 0.7,
    ):
        """
        Args:
            max_history: 最大历史记录数
            smooth_alpha: 平滑系数（新特征权重）
        """
        self.max_history = max_history
        self.smooth_alpha = smooth_alpha
        
        # 特征存储
        self.features = {}  # track_id -> current_feature
        self.history = {}   # track_id -> feature_history
        
    def update(
        self,
        track_id: int,
        new_feature: np.ndarray,
    ):
        """
        更新轨迹特征
        
        Args:
            track_id: 轨迹ID
            new_feature: 新提取的特征
        """
        if track_id in self.features:
            # 平滑更新
            old_feature = self.features[track_id]
            smoothed = self.smooth_alpha * new_feature + (1 - self.smooth_alpha) * old_feature
            self.features[track_id] = smoothed
        else:
            self.features[track_id] = new_feature.copy()
        
        # 记录历史
        if track_id not in self.history:
            self.history[track_id] = []
        self.history[track_id].append(new_feature.copy())
        
        # 限制历史长度
        if len(self.history[track_id]) > self.max_history:
            self.history[track_id] = self.history[track_id][-self.max_history:]
    
    def get(self, track_id: int) -> Optional[np.ndarray]:
        """获取轨迹特征"""
        return self.features.get(track_id)
    
    def get_history(self, track_id: int) -> List[np.ndarray]:
        """获取特征历史"""
        return self.history.get(track_id, [])
    
    def remove(self, track_id: int):
        """移除轨迹特征"""
        if track_id in self.features:
            del self.features[track_id]
        if track_id in self.history:
            del self.history[track_id]
    
    def check_consistency(
        self,
        track_id: int,
        new_feature: np.ndarray,
        threshold: float = 0.5,
    ) -> bool:
        """
        检查特征一致性
        
        Args:
            track_id: 轨迹ID
            new_feature: 新特征
            threshold: 一致性阈值
        
        Returns:
            is_consistent: 是否一致
        """
        if track_id not in self.features:
            return True
        
        current_feature = self.features[track_id]
        similarity = np.dot(new_feature, current_feature) / (
            np.linalg.norm(new_feature) * np.linalg.norm(current_feature) + 1e-8
        )
        
        return similarity > threshold
    
    def clear(self):
        """清除所有特征"""
        self.features.clear()
        self.history.clear()