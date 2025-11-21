import os
import os.path as osp
from glob import glob
from pathlib import Path
import inspect
import cv2
import numpy as np
from ..baseExtractor import baseVideoExtractor

import mediapipe as mp
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


class mediapipeExtractor(baseVideoExtractor):
    """
    Video feature extractor using MediaPipe. 
    Ref: https://mediapipe.dev/
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing MediaPipe video feature extractor...")
            logger.info(f"[MP file] {inspect.getfile(self.__class__)}")
            logger.info(f"[MP module __file__] {__file__}")
            super().__init__(config, logger)
            self.args = self.config['args']
            if self.args['visualize']: # drawing utilities
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                self.drawing_spec = self.mp_drawing.DrawingSpec(
                    thickness=1, circle_radius=1
                )
            if 'holistic' in self.args:
                self.kwargs = self.args['holistic']
                self.method = mp.solutions.holistic.Holistic
            elif 'face_mesh' in self.args:
                self.kwargs = self.args['face_mesh']
                self.kwargs['max_num_faces'] = 1
                self.method = mp.solutions.face_mesh.FaceMesh

            # [MOD] ImageEmbedder 준비(옵션이 주어질 때만 사용)
            # imageEmbedder path /home/ujeong/tmp/mobilenet_v3_small.tflite
            self._use_image_embedder = False
            self._image_embedder_options = None
            image_embedder_cfg = self.args.get('image_embedder', None)
            if image_embedder_cfg and image_embedder_cfg.get('model_path'):
                try:
                    BaseOptions = mp.tasks.BaseOptions
                    ImageEmbedder = mp.tasks.vision.ImageEmbedder
                    VisionRunningMode = mp.tasks.vision.RunningMode
                    ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
                    self._ImageEmbedder = ImageEmbedder
                    self._VisionRunningMode = VisionRunningMode
                    self._mpImage = mp.Image
                    self._ImageEmbedderOptions = ImageEmbedderOptions
                    self._BaseOptions = BaseOptions
                    self._image_embedder_options = ImageEmbedderOptions(
                        base_options=BaseOptions(model_asset_path=image_embedder_cfg['model_path']),
                        l2_normalize=image_embedder_cfg.get('l2_normalize', True),
                        quantize=image_embedder_cfg.get('quantize', False)
                    )
                    self._use_image_embedder = True
                except Exception as ee:
                    self.logger.warning(f"ImageEmbedder init skipped: {ee}")
                    self._use_image_embedder = False

        except Exception as e:
            self.logger.error("Failed to initialize mediapipeExtractor.")
            raise e

    # [MOD] 고정 길이로 자르거나 0-패딩하는 헬퍼
    def _fit_len(self, vec, target_len):
        if len(vec) >= target_len:
            return vec[:target_len]
        return vec + [0.0] * (target_len - len(vec))

    def extract(self, img_dir, video_name=None):
        """
        Function:
            Extract features from video file using MediaPipe.

        Parameters:
            img_dir: path to directory of images.
            video_name: video name used to save annotation images.

        Returns:
            video_result: extracted video features in numpy array.
        """
        try:
            video_result = []
            # [MOD] ImageEmbedder 인스턴스 (있을 때만)
            embedder = None
            if self._use_image_embedder:
                try:
                    embedder = self._ImageEmbedder.create_from_options(self._image_embedder_options)
                except Exception as ee:
                    self.logger.warning(f"Create ImageEmbedder failed: {ee}")
                    embedder = None

            with self.method(static_image_mode=False, **self.kwargs) as method:
                for image_path in sorted(glob(osp.join(img_dir, '*.bmp'))):
                    name = Path(image_path).stem
                    image_bgr = cv2.imread(image_path)
                    if image_bgr is None:
                        self.logger.debug(f"Failed to read {image_path}.")
                        continue
                    # 아주 어두운(거의 검은) 프레임은 스킵
                    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                    if float(np.mean(gray)) < 5.0 and float(np.max(gray)) < 10.0:
                        self.logger.debug(f"Black frame skipped: {image_path}")
                        continue
                    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                    result = method.process(image)

                    if 'holistic' in self.args:
                        # 존재 여부 체크
                        face_ok = bool(getattr(result, 'face_landmarks', None))
                        pose_ok = bool(getattr(result, 'pose_landmarks', None))
                        lh_ok = bool(getattr(result, 'left_hand_landmarks', None))
                        rh_ok = bool(getattr(result, 'right_hand_landmarks', None))

                        if not face_ok:
                            #self.logger.debug(f"No face detected in {image_path} (holistic).")
                            pass

                        if self.args['visualize']:
                            assert video_name is not None, \
                                "video_name should be passed in order to save annotation images."
                            annotated_image = image.copy()
                            if getattr(result, 'segmentation_mask', None) is not None:
                                condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1
                                bg_image = np.zeros(image.shape, dtype=np.uint8)
                                annotated_image = np.where(condition, annotated_image, bg_image)
                            if face_ok:
                                self.mp_drawing.draw_landmarks(
                                    image=annotated_image,
                                    landmark_list=result.face_landmarks,
                                    connections=mp.solutions.holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=self.mp_drawing_styles
                                    .get_default_face_mesh_tesselation_style()
                                )
                            if pose_ok:
                                self.mp_drawing.draw_landmarks(
                                    image=annotated_image,
                                    landmark_list=result.pose_landmarks,
                                    connections=mp.solutions.holistic.POSE_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing_styles.
                                    get_default_pose_landmarks_style()
                                )
                            if lh_ok:
                                self.mp_drawing.draw_landmarks(
                                    image=annotated_image,
                                    landmark_list=result.left_hand_landmarks,
                                    connections=mp.solutions.holistic.HAND_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing_styles.
                                    get_default_hand_landmarks_style()
                                )
                            if rh_ok:
                                self.mp_drawing.draw_landmarks(
                                    image=annotated_image,
                                    landmark_list=result.right_hand_landmarks,
                                    connections=mp.solutions.holistic.HAND_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing_styles.
                                    get_default_hand_landmarks_style()
                                )
                            os.makedirs(osp.join(self.args['visualize_dir'], video_name), exist_ok=True)
                            cv2.imwrite(osp.join(self.args['visualize_dir'], video_name, name + '.jpg'),
                                        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                        
                        res_image = []
                        if face_ok:
                            for lm in result.face_landmarks.landmark:
                                res_image.extend([lm.x, lm.y, lm.z])
                        else:
                            res_image.extend([0.0] * (468 * 3))
                        if pose_ok:
                            for lm in result.pose_landmarks.landmark:
                                res_image.extend([lm.x, lm.y, lm.z])
                        else:
                            res_image.extend([0.0] * (33 * 3))
                        if lh_ok:
                            for lm in result.left_hand_landmarks.landmark:
                                res_image.extend([lm.x, lm.y, lm.z])
                        else:
                            res_image.extend([0.0] * (21 * 3))
                        if rh_ok:
                            for lm in result.right_hand_landmarks.landmark:
                                res_image.extend([lm.x, lm.y, lm.z])
                        else:
                            res_image.extend([0.0] * (21 * 3))

                        # [MOD] 네 파트 모두 없으면 ImageEmbedder 임베딩으로 대체(길이는 1629 유지)
                        if (not face_ok) and (not pose_ok) and (not lh_ok) and (not rh_ok) and (embedder is not None):
                            try:
                                mp_image = self._mpImage(image_format=mp.ImageFormat.SRGB, data=image)
                                emb = embedder.embed(mp_image).embeddings[0].embedding
                                vec = list(emb)
                                res_image = self._fit_len(vec, 1629)  # 468+33+21+21 = 543, *3 = 1629
                            except Exception as ee:
                                self.logger.debug(f"ImageEmbedder fallback failed: {ee}")

                        video_result.append(res_image)

                    elif 'face_mesh' in self.args:
                        multi = getattr(result, 'multi_face_landmarks', None)
                        face_ok = bool(multi)
                        if not face_ok:
                            self.logger.debug(f"No face detected in {image_path} (face_mesh).")

                        if self.args['visualize']:
                            assert video_name is not None, \
                                "video_name should be passed in order to save annotation images."
                            annotated_image = image.copy()
                            if face_ok:
                                self.mp_drawing.draw_landmarks(
                                    image=annotated_image,
                                    landmark_list=multi[0],
                                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=self.mp_drawing_styles
                                    .get_default_face_mesh_tesselation_style()
                                )
                                self.mp_drawing.draw_landmarks(
                                    image=annotated_image,
                                    landmark_list=multi[0],
                                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=self.mp_drawing_styles
                                    .get_default_face_mesh_contours_style()
                                )
                            os.makedirs(osp.join(self.args['visualize_dir'], video_name), exist_ok=True)
                            cv2.imwrite(osp.join(self.args['visualize_dir'], video_name, name + '.jpg'),
                                        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                        res_image = []
                        if face_ok:
                            for lm in multi[0].landmark:
                                res_image.extend([lm.x, lm.y, lm.z])
                        else:
                            res_image.extend([0.0] * (468 * 3))

                        # [MOD] face 미검출이면 ImageEmbedder 임베딩으로 대체(길이는 468*3=1404 유지)
                        if (not face_ok) and (embedder is not None):
                            try:
                                mp_image = self._mpImage(image_format=mp.ImageFormat.SRGB, data=image)
                                emb = embedder.embed(mp_image).embeddings[0].embedding
                                vec = list(emb)
                                res_image = self._fit_len(vec, 1404)  # 468*3
                            except Exception as ee:
                                self.logger.debug(f"ImageEmbedder fallback failed: {ee}")

                        video_result.append(res_image)

                video_result = np.array(video_result)
                return video_result

        except Exception as e:
            self.logger.error(f"Failed to extract video features with MediaPipe from {video_name}.")
            raise e
