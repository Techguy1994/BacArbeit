import numpy as np

tf_np = np.load("output_tf.npy")
pyarmnn_np = np.load("output_pyarmnn.npy")

print("Shape of tf", tf_np.shape)
print("Shaper of pyarmmn", pyarmnn_np.shape)

print((tf_np==pyarmnn_np).all())
print(np.allclose(tf_np, pyarmnn_np))

if (tf_np==pyarmnn_np).all():
    print("Same")

print(tf_np)

print("\n")

print(pyarmnn_np)
