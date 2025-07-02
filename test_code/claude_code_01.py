#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Florida Wildlife Camera Trap Dataset - Complete CNN Implementation
macOS VSCode 환경용 완전한 구현 코드

필요한 라이브러리 설치:
pip install tensorflow-macos tensorflow-metal
pip install opencv-python scikit-learn matplotlib seaborn pandas
pip install pillow tqdm
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import cv2
import os
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU 설정 (Apple Silicon 최적화)
print("TensorFlow 버전:", tf.__version__)
print("사용 가능한 GPU:", tf.config.list_physical_devices('GPU'))

# ========================== 데이터셋 구조 시뮬레이션 ==========================
class FloridaWildlifeDatasetSimulator:
    """
    실제 데이터셋이 없을 경우를 위한 데이터 구조 시뮬레이터
    실제 데이터가 있다면 이 부분을 수정하세요
    """
    def __init__(self, base_dir='florida_wildlife_dataset'):
        self.base_dir = Path(base_dir)
        
        # 22개 종 클래스 (실제 데이터셋 기준)
        self.class_names = [
            'cattle', 'wild_pig', 'white_tailed_deer', 'raccoon', 'bird',
            'opossum', 'rabbit', 'squirrel', 'bobcat', 'chicken', 'horse',
            'crow', 'turkey', 'alligator', 'armadillo', 'otter', 'dog',
            'coyote', 'bear', 'cat', 'florida_panther', 'unknown'
        ]
        
        # 클래스별 이미지 수 (불균형 시뮬레이션)
        self.class_distribution = {
            'cattle': 15000, 'wild_pig': 12000, 'white_tailed_deer': 18000,
            'raccoon': 8000, 'bird': 6000, 'opossum': 5000, 'rabbit': 4000,
            'squirrel': 3500, 'bobcat': 3000, 'chicken': 2500, 'horse': 4500,
            'crow': 2000, 'turkey': 3500, 'alligator': 1500, 'armadillo': 2500,
            'otter': 1000, 'dog': 2000, 'coyote': 1500, 'bear': 800,
            'cat': 1200, 'florida_panther': 2500, 'unknown': 3995
        }
        
    def create_dummy_dataset(self):
        """더미 데이터셋 구조 생성 (실제 이미지 없이)"""
        print("데이터셋 구조 생성 중...")
        
        # 디렉토리 생성
        for split in ['train', 'val', 'test']:
            for class_name in self.class_names:
                (self.base_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 생성
        metadata = {
            'class_names': self.class_names,
            'class_distribution': self.class_distribution,
            'total_images': sum(self.class_distribution.values()),
            'created_date': datetime.now().isoformat()
        }
        
        with open(self.base_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"데이터셋 구조 생성 완료: {self.base_dir}")
        return metadata

# ========================== 데이터 로더 ==========================
class WildlifeDataLoader:
    """Florida Wildlife 데이터셋 로더"""
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.AUTOTUNE = tf.data.AUTOTUNE
        
        # 메타데이터 로드
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata['class_names']
        self.num_classes = len(self.class_names)
        
    def create_dataset(self, subset='train', shuffle=True, augment=True):
        """TensorFlow 데이터셋 생성"""
        
        # 실제 이미지가 있는 경우의 데이터셋 생성
        if (self.data_dir / subset).exists() and any((self.data_dir / subset).iterdir()):
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_dir / subset,
                labels='inferred',
                label_mode='categorical',
                class_names=self.class_names,
                color_mode='rgb',
                batch_size=self.batch_size,
                image_size=self.img_size,
                shuffle=shuffle,
                seed=42
            )
        else:
            # 더미 데이터셋 생성 (테스트용)
            print(f"경고: {subset} 데이터가 없습니다. 더미 데이터를 생성합니다.")
            dataset = self._create_dummy_dataset(subset)
        
        # 데이터 증강 파이프라인 적용
        if augment and subset == 'train':
            dataset = dataset.map(self._augment, num_parallel_calls=self.AUTOTUNE)
        
        # 성능 최적화
        dataset = dataset.prefetch(self.AUTOTUNE)
        
        return dataset
    
    def _create_dummy_dataset(self, subset):
        """더미 데이터셋 생성 (실제 이미지 없이 테스트용)"""
        # 각 클래스별 샘플 수 계산
        samples_per_class = {
            'train': 100,
            'val': 20,
            'test': 30
        }[subset]
        
        # 더미 이미지와 레이블 생성
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            # 랜덤 노이즈 이미지 생성
            class_images = tf.random.normal((samples_per_class, *self.img_size, 3))
            class_labels = tf.one_hot([class_idx] * samples_per_class, self.num_classes)
            
            images.append(class_images)
            labels.append(class_labels)
        
        # 결합 및 셔플
        images = tf.concat(images, axis=0)
        labels = tf.concat(labels, axis=0)
        
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.batch(self.batch_size)
        
        return dataset
    
    def _augment(self, image, label):
        """데이터 증강 파이프라인"""
        # 랜덤 수평 뒤집기
        image = tf.image.random_flip_left_right(image)
        
        # 랜덤 회전 (-20도 ~ +20도)
        image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # 랜덤 밝기 조정 (±20%)
        image = tf.image.random_brightness(image, 0.2)
        
        # 랜덤 대비 조정 (0.8 ~ 1.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # 랜덤 채도 조정 (0.8 ~ 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        # 랜덤 색조 조정
        image = tf.image.random_hue(image, 0.1)
        
        # 모션 블러 시뮬레이션 (카메라 트랩 특성)
        if tf.random.uniform([]) > 0.7:
            image = self._add_motion_blur(image)
        
        # 값 범위 클리핑
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def _add_motion_blur(self, image):
        """모션 블러 효과 추가 (야생동물 움직임 시뮬레이션)"""
        # 간단한 모션 블러 구현
        kernel_size = tf.random.uniform([], 3, 7, dtype=tf.int32)
        # TensorFlow ops로 구현하기 복잡하므로 가우시안 블러로 대체
        image = tf.nn.avg_pool2d(
            tf.expand_dims(image, 0),
            ksize=[kernel_size, kernel_size],
            strides=[1, 1],
            padding='SAME'
        )
        return tf.squeeze(image, 0)
    
    def calculate_class_weights(self):
        """클래스 가중치 계산 (불균형 처리)"""
        class_counts = list(self.metadata['class_distribution'].values())
        total_samples = sum(class_counts)
        
        class_weights = {}
        for i, count in enumerate(class_counts):
            # 역 빈도 가중치
            weight = total_samples / (len(class_counts) * count)
            # 멸종위기종(플로리다 팬서) 추가 가중치
            if self.class_names[i] == 'florida_panther':
                weight *= 2.0
            class_weights[i] = weight
        
        return class_weights

# ========================== 모델 아키텍처 ==========================
class WildlifeCNNModel:
    """야생동물 분류를 위한 CNN 모델"""
    
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def build_resnet50_transfer(self):
        """ResNet50 전이학습 모델 (논문 baseline)"""
        # 사전 훈련된 ResNet50 로드
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # 초기에는 사전훈련 가중치 고정
        base_model.trainable = False
        
        # 모델 구성
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # 전처리 레이어
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        
        # 기본 모델
        x = base_model(x, training=False)
        
        # 분류 헤드
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name='wildlife_resnet50')
        
        return model, base_model
    
    def build_efficientnet_b3(self):
        """EfficientNet-B3 모델 (더 나은 성능)"""
        base_model = tf.keras.applications.EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x = base_model(x, training=False)
        
        # 분류 헤드
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name='wildlife_efficientnet_b3')
        
        return model, base_model
    
    def build_ensemble_model(self):
        """앙상블 모델 (ResNet50 + EfficientNet-B3)"""
        # ResNet50 브랜치
        resnet_base = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        resnet_base.trainable = False
        
        # EfficientNet-B3 브랜치
        efficientnet_base = tf.keras.applications.EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        efficientnet_base.trainable = False
        
        # 입력 레이어
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # ResNet50 경로
        x1 = tf.keras.applications.resnet50.preprocess_input(inputs)
        x1 = resnet_base(x1, training=False)
        x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
        
        # EfficientNet-B3 경로
        x2 = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x2 = efficientnet_base(x2, training=False)
        x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
        
        # 특징 결합
        combined = tf.keras.layers.concatenate([x1, x2])
        
        # 분류 헤드
        x = tf.keras.layers.Dense(1024, activation='relu')(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name='wildlife_ensemble')
        
        return model, [resnet_base, efficientnet_base]

# ========================== 커스텀 손실 함수 ==========================
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss - 클래스 불균형 처리에 효과적
    멸종위기종 검출 성능 향상
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # 크로스 엔트로피 계산
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -tf.math.log(p_t)
        
        # Focal loss 가중치
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return focal_loss_fixed

# ========================== 학습 관리자 ==========================
class WildlifeTrainer:
    """모델 학습 및 평가 관리"""
    
    def __init__(self, model, class_names, save_dir='model_checkpoints'):
        self.model = model
        self.class_names = class_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.history = None
        
    def compile_model(self, learning_rate=1e-3, loss='categorical_crossentropy', 
                     class_weights=None):
        """모델 컴파일"""
        
        # 멸종위기종 중심 메트릭
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),  # 멸종위기종에 중요
            tf.keras.metrics.AUC(name='auc')
        ]
        
        # 옵티마이저
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # 컴파일
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.class_weights = class_weights
        
    def create_callbacks(self):
        """학습 콜백 생성"""
        callbacks_list = [
            # 모델 체크포인트
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.save_dir / 'best_model.h5'),
                monitor='val_recall',  # 리콜 기준 저장
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # 조기 종료
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # 학습률 감소
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV 로거
            tf.keras.callbacks.CSVLogger(
                str(self.save_dir / 'training_log.csv')
            ),
            
            # 커스텀 콜백 - 멸종위기종 성능 모니터링
            EndangeredSpeciesCallback(
                self.class_names,
                endangered_species=['florida_panther', 'bear']
            )
        ]
        
        return callbacks_list
    
    def train(self, train_dataset, val_dataset, epochs=100, fine_tune_at=None):
        """모델 학습"""
        
        print("="*50)
        print("모델 학습 시작")
        print("="*50)
        
        callbacks = self.create_callbacks()
        
        # 초기 학습 (전이학습)
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs//2 if fine_tune_at else epochs,
            callbacks=callbacks,
            class_weight=self.class_weights
        )
        
        # 파인튜닝 (선택적)
        if fine_tune_at is not None:
            print("\n파인튜닝 시작...")
            self._fine_tune_model(fine_tune_at)
            
            # 파인튜닝 학습
            fine_tune_history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                initial_epoch=len(self.history.history['loss']),
                callbacks=callbacks,
                class_weight=self.class_weights
            )
            
            # 히스토리 병합
            for key in self.history.history:
                self.history.history[key].extend(fine_tune_history.history[key])
        
        return self.history
    
    def _fine_tune_model(self, fine_tune_at):
        """모델 파인튜닝 설정"""
        # 기본 모델 찾기
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
        
        if base_model:
            base_model.trainable = True
            
            # fine_tune_at 레이어부터 학습 가능하게 설정
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            # 낮은 학습률로 재컴파일
            self.compile_model(
                learning_rate=1e-5,
                loss=focal_loss(),
                class_weights=self.class_weights
            )
    
    def evaluate(self, test_dataset):
        """모델 평가"""
        print("\n모델 평가 중...")
        
        # 기본 평가
        test_loss, test_acc, test_top3_acc, test_prec, test_recall, test_auc = \
            self.model.evaluate(test_dataset, verbose=1)
        
        # 예측 수집
        y_true = []
        y_pred = []
        
        for images, labels in test_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        # 분류 리포트
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_top3_accuracy': test_top3_acc,
            'test_precision': test_prec,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        # 결과 저장
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

# ========================== 커스텀 콜백 ==========================
class EndangeredSpeciesCallback(tf.keras.callbacks.Callback):
    """멸종위기종 검출 성능 모니터링"""
    
    def __init__(self, class_names, endangered_species):
        super().__init__()
        self.class_names = class_names
        self.endangered_species = endangered_species
        self.endangered_indices = [
            class_names.index(species) 
            for species in endangered_species 
            if species in class_names
        ]
    
    def on_epoch_end(self, epoch, logs=None):
        """에폭 종료 시 멸종위기종 성능 출력"""
        if epoch % 5 == 0:  # 5 에폭마다
            print(f"\n멸종위기종 모니터링 (Epoch {epoch+1}):")
            print(f"전체 Recall: {logs.get('val_recall', 0):.4f}")
            # 실제 구현 시 클래스별 성능 계산 추가

# ========================== 시각화 도구 ==========================
class WildlifeVisualizer:
    """학습 결과 시각화"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        
    def plot_training_history(self, history, save_path='training_history.png'):
        """학습 히스토리 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 정확도
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 손실
        axes[0, 1].plot(history.history['loss'], label='Train')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-3 정확도
        if 'top_3_accuracy' in history.history:
            axes[0, 2].plot(history.history['top_3_accuracy'], label='Train')
            axes[0, 2].plot(history.history['val_top_3_accuracy'], label='Validation')
            axes[0, 2].set_title('Top-3 Accuracy', fontsize=14)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Top-3 Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall (멸종위기종에 중요)
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall (Important for Endangered Species)', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # AUC
        axes[1, 2].plot(history.history['auc'], label='Train')
        axes[1, 2].plot(history.history['val_auc'], label='Validation')
        axes[1, 2].set_title('AUC', fontsize=14)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AUC')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png', normalize=True):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(20, 16))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # 히트맵 그리기
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            square=True,
            cbar_kws={'label': 'Prediction Probability' if normalize else 'Count'}
        )
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 멸종위기종 강조
        endangered_indices = [
            self.class_names.index('florida_panther'),
            self.class_names.index('bear')
        ]
        for idx in endangered_indices:
            rect = plt.Rectangle((idx-0.5, idx-0.5), 1, 1, 
                               fill=False, edgecolor='red', lw=3)
            plt.gca().add_patch(rect)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_performance(self, classification_report, save_path='class_performance.png'):
        """클래스별 성능 시각화"""
        # 데이터 준비
        metrics = ['precision', 'recall', 'f1-score']
        class_data = []
        
        for class_name in self.class_names:
            if class_name in classification_report:
                class_metrics = classification_report[class_name]
                class_data.append([
                    class_metrics['precision'],
                    class_metrics['recall'],
                    class_metrics['f1-score']
                ])
            else:
                class_data.append([0, 0, 0])
        
        class_data = np.array(class_data)
        
        # 히트맵 생성
        plt.figure(figsize=(10, 12))
        sns.heatmap(
            class_data,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=metrics,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Score'},
            vmin=0,
            vmax=1
        )
        
        plt.title('Per-Class Performance Metrics', fontsize=16, pad=20)
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Species', fontsize=14)
        
        # 멸종위기종 강조
        endangered_species = ['florida_panther', 'bear']
        for species in endangered_species:
            if species in self.class_names:
                idx = self.class_names.index(species)
                plt.text(-0.5, idx + 0.5, '⚠️', fontsize=20, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ========================== 메인 실행 함수 ==========================
def main():
    """전체 파이프라인 실행"""
    
    print("="*60)
    print("Florida Wildlife Camera Trap CNN 시작")
    print("="*60)
    
    # 1. 데이터셋 준비
    print("\n1. 데이터셋 준비 중...")
    simulator = FloridaWildlifeDatasetSimulator()
    metadata = simulator.create_dummy_dataset()
    
    # 2. 데이터 로더 생성
    print("\n2. 데이터 로더 초기화...")
    data_loader = WildlifeDataLoader(
        data_dir='florida_wildlife_dataset',
        img_size=(224, 224),
        batch_size=32
    )
    
    # 데이터셋 생성
    train_dataset = data_loader.create_dataset('train', shuffle=True, augment=True)
    val_dataset = data_loader.create_dataset('val', shuffle=False, augment=False)
    test_dataset = data_loader.create_dataset('test', shuffle=False, augment=False)
    
    # 클래스 가중치 계산
    class_weights = data_loader.calculate_class_weights()
    
    # 3. 모델 생성
    print("\n3. 모델 구축 중...")
    model_builder = WildlifeCNNModel(
        num_classes=len(data_loader.class_names),
        input_shape=(224, 224, 3)
    )
    
    # ResNet50 모델 사용 (논문 baseline)
    model, base_model = model_builder.build_resnet50_transfer()
    
    # 모델 요약 출력
    print("\n모델 아키텍처:")
    model.summary()
    
    # 4. 학습 준비
    print("\n4. 학습 준비...")
    trainer = WildlifeTrainer(
        model=model,
        class_names=data_loader.class_names,
        save_dir='model_checkpoints'
    )
    
    # 모델 컴파일
    trainer.compile_model(
        learning_rate=1e-3,
        loss=focal_loss(),  # Focal loss 사용
        class_weights=class_weights
    )
    
    # 5. 모델 학습
    print("\n5. 모델 학습 시작...")
    history = trainer.train(
        train_dataset,
        val_dataset,
        epochs=50,  # 실제로는 100+ 권장
        fine_tune_at=100  # ResNet50의 100번째 레이어부터 파인튜닝
    )
    
    # 6. 모델 평가
    print("\n6. 모델 평가...")
    results = trainer.evaluate(test_dataset)
    
    print(f"\n최종 테스트 성능:")
    print(f"정확도: {results['test_accuracy']:.4f}")
    print(f"Top-3 정확도: {results['test_top3_accuracy']:.4f}")
    print(f"Precision: {results['test_precision']:.4f}")
    print(f"Recall: {results['test_recall']:.4f}")
    print(f"AUC: {results['test_auc']:.4f}")
    
    # 7. 시각화
    print("\n7. 결과 시각화...")
    visualizer = WildlifeVisualizer(data_loader.class_names)
    
    # 학습 히스토리
    visualizer.plot_training_history(history)
    
    # 혼동 행렬
    visualizer.plot_confusion_matrix(
        results['confusion_matrix'],
        normalize=True
    )
    
    # 클래스별 성능
    visualizer.plot_class_performance(
        results['classification_report']
    )
    
    # 8. 모델 저장
    print("\n8. 최종 모델 저장...")
    model.save('florida_wildlife_final_model.h5')
    
    # TensorFlow Lite 변환 (모바일 배포용)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('florida_wildlife_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("\n완료! 모든 결과가 저장되었습니다.")
    print("="*60)

# ========================== 추론 도구 ==========================
class WildlifePredictor:
    """학습된 모델로 새로운 이미지 예측"""
    
    def __init__(self, model_path, class_names):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss()}
        )
        self.class_names = class_names
        
    def predict_image(self, image_path, top_k=3):
        """단일 이미지 예측"""
        # 이미지 로드 및 전처리
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        # 예측
        predictions = self.model.predict(img_array)[0]
        
        # Top-K 예측
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'species': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'is_endangered': self.class_names[idx] in ['florida_panther', 'bear']
            })
        
        return results
    
    def batch_predict(self, image_dir, output_csv='predictions.csv'):
        """디렉토리의 모든 이미지 예측"""
        results = []
        
        image_paths = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.png'))
        
        print(f"{len(image_paths)}개 이미지 처리 중...")
        
        for img_path in tqdm(image_paths):
            predictions = self.predict_image(img_path)
            
            results.append({
                'filename': img_path.name,
                'predicted_species': predictions[0]['species'],
                'confidence': predictions[0]['confidence'],
                'is_endangered': predictions[0]['is_endangered'],
                'top_3_predictions': str(predictions[:3])
            })
        
        # CSV로 저장
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        # 멸종위기종 알림
        endangered_detections = df[df['is_endangered'] == True]
        if len(endangered_detections) > 0:
            print(f"\n⚠️  경고: {len(endangered_detections)}개의 멸종위기종이 감지되었습니다!")
            print(endangered_detections[['filename', 'predicted_species', 'confidence']])
        
        return df

# ========================== 실행 ==========================
if __name__ == "__main__":
    # GPU 메모리 증가 허용 (Apple Silicon)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # 메인 파이프라인 실행
    main()
    
    # 예측 예시 (학습 완료 후)
    # predictor = WildlifePredictor(
    #     'florida_wildlife_final_model.h5',
    #     class_names=metadata['class_names']
    # )
    # results = predictor.predict_image('test_image.jpg')
    # print(results)