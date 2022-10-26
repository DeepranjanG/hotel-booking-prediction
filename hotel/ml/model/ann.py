# import tensorflow as tf
# from hotel.logger import logging
# from hotel.entity.config_entity import AnnModelTrainerConfig
# from hotel.utils.main_utils import read_yaml_file
# from hotel.constant.training_pipeline import MODEL_FILE_PATH
# class AnnModel:
#     def __init__(self,ann_model_trainer_config: AnnModelTrainerConfig):
#         """
#         :param data_transformation_config: Configuration for ann model trainer
#         """
#         self.config = ann_model_trainer_config
#         self._model_config = read_yaml_file(file_path=MODEL_FILE_PATH)
#
#
#     def create_ann_model(self):
#         """
#         INPUT_SIZE: number of independent features
#         DENSE_LAYER: Units of neuron in each layer
#         OUTPUT: if it's a binary classification then 1 else number of classes in the target column.
#         Returns:
#             Simple sequential model
#
#         -------
#
#         """
#         model = tf.keras.Sequential([
#             tf.keras.Input(self._model_config['INPUT_SIZE']),
#             tf.keras.layers.Dense(self._model_config['DENSE_LAYER'], activation="relu"),
#             tf.keras.layers.Dense(self._model_config['DENSE_LAYER'], activation="relu"),
#             tf.keras.layers.Dense(self._model_config['OUTPUT'], activation="sigmoid"),
#         ])
#
#         return model
#
#
#     def early_stopping_callback(self):
#
#         early_stopping = tf.keras.callbacks.EarlyStopping(patience=self._model_config['PATIENCE'],
#                                          restore_best_weights=self._model_config['RESTORE_BEST_WEIGHTS']
#         )
#
#         return early_stopping
#
#
#     def model_fit(self, x, y):
#
#         logging.info("Loading ANN Model")
#
#         model = self.create_ann_model()
#
#         logging.info("Model Compiling")
#
#         model.compile(
#             optimizer=self._model_config['OPTIMIZER'],
#             loss=self._model_config['LOSS'],
#             metrics=["accuracy"]
#         )
#
#         logging.info("Fitting model with train and test dataset")
#
#         model.fit(x, y,
#             validation_split=self._model_config['VALIDATION_SPLIT'],
#             batch_size=self._model_config['BATCH_SIZE'],
#             epochs=self._model_config['EPOCHS'],
#             callbacks=self.early_stopping_callback()
#             )
#
#         logging.info("Training Completed")
#
#
#
