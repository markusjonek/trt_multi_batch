"""
    Class to perform detections using TensorRT.
"""
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class HostDeviceMem(object):
    """ Helper data class. """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtDetector(object):
    def __init__(self, engine_path, img_size, batch_size):
        """ Init function
            @param engine_path:  Path to the TensorRT serialised engine
            @param img_size:     Size of each image dimension
        """
        self.TRT_LOGGER = trt.Logger()
        self.img_size = img_size

        self.engine = self.get_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.buffers = self.allocate_buffers(batch_size=1)

        self.context.set_binding_shape(1, (batch_size, 3, self.img_size, self.img_size))

    def get_engine(self, engine_path):
        """ Load serialised engine from file """
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self, batch_size):
        """ Allocate necessary buffers for inference on the GPU
            @param batch_size: Size of the batches
            @return 
                - inputs: buffer for inputs  
                - outputs: buffer for outputs
                - bindings: device bindings
                - stream: GPU stream, sequence of operations
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings
            bindings.append(int(device_mem))
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


    def do_inference(self, bindings, inputs, outputs, stream):
        """ Inference on the GPU """
        # Transfer input data to the GPU
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs
        return [out.host for out in outputs]


    def detect(self, imgs):

        inputs, outputs, bindings, stream = self.buffers

        inputs[0].host = imgs.ravel()

        trt_outputs = self.do_inference(
            bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        return trt_outputs[0]



if __name__ == "__main__":
    detector = TrtDetector("./engine_batch32.trt", 80, 32)

    random_imgs = np.random.rand(32, 3, 80, 80).astype('float16').ravel()

    keys_batches = detector.detect(random_imgs)

    keys_batches = np.reshape(keys_batches, (32, 14))

    print(keys_batches)
