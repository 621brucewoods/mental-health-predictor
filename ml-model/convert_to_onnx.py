import tensorflow as tf
import tf2onnx

# Load the Keras model
model = tf.keras.models.load_model("models/mental_health_model.pkl")

# Convert the Keras model to ONNX
input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

# Save the ONNX model
with open("models/mental_health_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model successfully converted to ONNX and saved as 'mental_health_model.onnx'.")