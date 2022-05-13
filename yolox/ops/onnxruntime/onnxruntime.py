import os
import glob
onnxruntime_path = os.path.dirname(os.path.realpath(__file__))

__all__ = ["get_onnxruntime_custom_ops_path"]

def get_onnxruntime_custom_ops_path():
    type = ("so", )
    dynamic_list = []
    for t in type:
        for p in glob.glob(onnxruntime_path + "/**/*." + t, recursive=True):
            dynamic_list.append(p)
    return dynamic_list 

if __name__ == "__main__":
    get_onnxruntime_custom_ops_path()