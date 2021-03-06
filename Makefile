GPU=1
CUDNN=1
CUDNN_HALF=0
OPENCV=0
AVX=0
OPENMP=0
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
USE_CPP=1
DEBUG=1
USE_PROTOBUF=1

# set GPU=1 and CUDNN=1 to speedup on GPU
# # set CUDNN_HALF=1 to further speedup 3 x times (Mixed-precision on Tensor Cores) GPU: Volta, Xavier, Turing and higher
# # set AVX=1 and OPENMP=1 to speedup on CPU (if error occurs then set AVX=0)

ARCH= -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_61,code=[sm_61,compute_61]

OS := $(shell uname)

# GeForce RTX 3070, 3080, 3090
# # ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]
#
# # Kepler GeForce GTX 770, GTX 760, GT 740
# # ARCH= -gencode arch=compute_30,code=sm_30
#
# # Tesla A100 (GA100), DGX-A100, RTX 3080
# # ARCH= -gencode arch=compute_80,code=[sm_80,compute_80]
#
# # Tesla V100
# # ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]
#
# # GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
# # ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]
#
# # Jetson XAVIER
# # ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]
#
# # GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# # ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
#
# # GP100/Tesla P100 - DGX-1
# # ARCH= -gencode arch=compute_60,code=sm_60
#
# # For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
# # ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]
#
# # For Jetson Tx2 or Drive-PX2 uncomment:
# # ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]
#
# # For Tesla GA10x cards, RTX 3090, RTX 3080, RTX 3070, RTX A6000, RTX A40 uncomment:
# # ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]

EXEC=tritonserver
OBJDIR=./obj/
VPATH=./src

ifeq ($(LIBSO), 1)
LIBNAMESO=libBOE_trison_server.so
APPNAMESO=uselib
endif

ifeq ($(USE_CPP), 1)
CC=g++
else
CC=gcc
endif

CPP=g++
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -ldl -lrt -lre2 -lb64 -lm -pthread -levhtp -levent
#LDFLAGS= -lX11 -ldl -lm -pthread -I/usr/include/boost 
COMMON= -Iinclude/ -I3rd/ -I3rd/libevent/include -I3rd/libevhtp/include -I/usr/include/boost -L/opt/work/boe/tritonserver/3rd/libevhtp/lib -L/opt/work/triton-server/build/libevent/install/lib 
CFLAGS=-Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -fPIC

ifeq ($(DEBUG), 1)
OPTS= -O0 -g
#OPTS= -Og -g
COMMON+= -DDEBUG
CFLAGS+= -DDEBUG
else
ifeq ($(AVX), 1)
CFLAGS+= -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a
endif
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv`
COMMON+= `pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv`
endif

ifeq ($(USE_PROTOBUF),1)
COMMON+= -DPROTOBUF
CFLAGS+= -DPROTOBUF
LDFLAGS+= `pkg-config --libs protobuf`
endif

ifeq ($(OPENMP), 1)
ifeq ($(OS),Darwin) #MAC
CFLAGS+= -Xpreprocessor -fopenmp
else
CFLAGS+= -fopenmp
endif
LDFLAGS+= -lgomp
endif

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/ -I/usr/local/include
CFLAGS+= -DGPU
ifeq ($(OS),Darwin) #MAC
LDFLAGS+= -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand
else
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif
endif

ifeq ($(CUDNN), 1)
COMMON+= -DCUDNN
ifeq ($(OS),Darwin) #MAC
CFLAGS+= -DCUDNN -I/usr/local/cuda/include
LDFLAGS+= -L/usr/local/cuda/lib -lcudnn
else
CFLAGS+= -DCUDNN -I/usr/local/cudnn/include
LDFLAGS+= -L/usr/local/cudnn/lib64 -lcudnn
endif
endif

ifeq ($(CUDNN_HALF), 1)
COMMON+= -DCUDNN_HALF
CFLAGS+= -DCUDNN_HALF
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70]
endif

OBJ=main.o logging.o tritonserver.o status.o model_config.o model_config.pb.o model_repository_manager.o filesystem.o pytorchautofill.o model_config_utils.o cuda_utils.o autofill.o \
    server.o cuda_memory_manager.o cnmem.o async_work_queue.o pinned_memory_manager.o persistent_backend_manager.o triton_backend_manager.o triton_memory_manager.o \
    table_printer.o infer_request.o memory.o dynamic_batch_scheduler.o sequence_batch_scheduler.o scheduler_utils.o infer_reponse.o backend.o label_provider.o triton_model.o \
    triton_backend_config.o triton_model_instance.o infer_parameter.o http_server.o shared_memory_manager.o common.o classification.o onnxautofill.o
ifeq ($(GPU), 1)
#LDFLAGS+= -lstdc++
#OBJ+=convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
#DEPS = $(wildcard src/*.h) Makefile include/net.h

#all: $(OBJDIR) $(EXEC) $(LIBNAMESO)
all: $(OBJDIR) backup setchmod results $(LIBNAMESO) $(APPNAMESO)

ifeq ($(LIBSO), 1)
CFLAGS+= -fPIC

$(LIBNAMESO): $(OBJDIR) $(OBJS)
	$(CPP) -shared -std=c++11 -fvisibility=hidden -DLIB_EXPORTS $(COMMON) $(OBJS) $(CFLAGS) -o $@ $(LDFLAGS)

$(APPNAMESO): $(LIBNAMESO)
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) -o $@ $(LDFLAGS) -L ./ -l:$(LIBNAMESO)
endif

$(EXEC): $(OBJS)
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj/pytorch/%.o: %.cc $(DEPS)
	        $(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cc $(DEPS)
	        $(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cpp #$(DEPS)
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

backup:
	mkdir -p backup
results:
	mkdir -p results
setchmod:
	chmod +x *.sh

.PHONY: clean

clean:
	rm -rf ./core $(OBJS) $(LIBNAMESO) $(APPNAMESO)
