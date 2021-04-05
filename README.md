read dates from `HMEq_FFTW` and feed it to TensorFlow custom model.
usage:
1. call read.py to pick up data calculated by HMEq_FFTW.
`python3 read.py`
2. sampling down original data to compose input to ML model.
`python3 sampling_down.py`
3. build up TFRecord files.
`python3 data2tfrecord.py`
4. execute ML learning task.
`python3 main_ResNet.py`
