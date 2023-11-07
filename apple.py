!pip install onnxruntime
!pip install skl2onnx
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
random_state=42)

initial_types = [('input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_types)
onnx.save_model(onnx_model, 'iris_model.onnx')
import onnx
import onnxruntime as ort
import numpy as np

providers = ['CPUExecutionProvider']
ort_session = ort.InferenceSession("iris_model.onnx", providers=providers)

input_data = np.array([[5.1, 3.5, 1.4, 0.2],[6.3, 2.8, 5.1, 1.5]], dtype=np.float32)

predictions = ort_session.run(None, {"input": input_data})
print("Predictions:", predictions)
