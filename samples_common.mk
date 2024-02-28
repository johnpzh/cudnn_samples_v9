#Setting SMS for all samples
#architecture

CUDA_VERSION := $(shell cat $(CUDA_PATH)/include/cuda.h |grep "define CUDA_VERSION" |awk '{print $$3}')
$(info CUDA_VERSION is ${CUDA_VERSION})
#Link against cublasLt for CUDA 10.1 and up.
CUBLASLT:=false
ifeq ($(shell test $(CUDA_VERSION) -ge 10010; echo $$?),0)
CUBLASLT:=true
endif
$(info Linking agains cublasLt = $(CUBLASLT))

ifeq ($(CUDA_VERSION),8000)
SMS_VOLTA =
else
ifneq ($(TARGET_ARCH), ppc64le)
ifeq ($(CUDA_VERSION), $(filter $(CUDA_VERSION), 9000 9010 9020))
SMS_VOLTA ?= 70
else
ifeq ($(TARGET_OS), darwin)
SMS_VOLTA ?= 70
else
SMS_VOLTA ?= 70 72 75
endif #ifneq ($(TARGET_OS), darwin)
endif #ifeq ($(CUDA_VERSION), $(filter $(CUDA_VERSION), 9000 9010 9020))
else
SMS_VOLTA ?= 70
endif #ifneq ($(TARGET_ARCH), ppc64le)
endif #ifeq ($(CUDA_VERSION),8000 )

SMS_AMPERE =
ifeq ($(shell expr $(CUDA_VERSION) \>= 11000),1)
SMS_AMPERE = 80
endif

ifeq ($(shell expr $(CUDA_VERSION) \>= 11010),1)
SMS_AMPERE += 86
endif

ifeq ($(shell expr $(CUDA_VERSION) \>= 11042),1)
SMS_AMPERE += 87
endif

SMS_HOPPER =
ifeq ($(shell expr $(CUDA_VERSION) \>= 11080),1)
SMS_HOPPER += 90
endif
 
ifeq ($(shell expr $(CUDA_VERSION) \>= 12000),1)
SMS_LEGACY ?= 50 53 60 61 62
else
SMS_LEGACY ?= 35 50 53 60 61 62
endif

SMS ?= $(SMS_LEGACY) $(SMS_VOLTA) $(SMS_AMPERE) $(SMS_HOPPER)
