# Acknowledgement: Functionality for creating make rules of dependencies is
# based on code presented here <http://codereview.stackexchange.com/q/11109>

DIS_HOME := /opt/DIS
CU_HOME	 := /usr/local/cuda

# Locate project files
CLIENT_SRC	 := c63enc.cu motion.cu cuda_dct.cu tables.cu io.cu c63_write.cu me.cu sisci.c
#SERVER_SRC	 := read_write_machine/c63enc.c read_write_machine/dsp.c read_write_machine/tables.c io.c read_write_machine/c63_write.c read_write_machine/common.c read_write_machine/me.c
HEADERS		 := common.h #testlib.h 
CLIENT_OBJ 	 := $(CLIENT_SRC:%.c=%.o) #$(DIS_HOME)/src/testlib/testlib.o
#SERVER_OBJ	 := $(SERVER_SRC:%.c=%.o)

# Add dot prefix to hide files.
#SERVER_DEPS  := $(addprefix .,$(SERVER_SRC:.c=.d))
CLIENT_DEPS  := $(addprefix .,$(CLIENT_SRC:.c=.d))

PROGRAMS     := c63enc

# Compiler and linker settings
CC	     := /usr/bin/gcc
NVCC 	 := $(CU_HOME)/bin/nvcc
INCLUDE	 := -I$(DIS_HOME)/include -I$(DIS_HOME)/include/dis -I $(DIS_HOME)/src/include -I$(CU_HOME)/include
CFLAGS	 := -std=c++11 -D_REENTRANT -O0 $(INCLUDE) -DOS_IS_LINUX
LDLIBS	 := -lsisci

# Use the compiler to generate make rules. See gcc manual for details.
MFLAGS   := -MMD -MP -MF

ifneq ($(shell getconf LONG_BIT),)
	LDFLAGS := -L$(DIS_HOME)/lib64 -L$(CU_HOME)/lib64 $(LDLIBS)
else
	LDFLAGS := -L$(DIS_HOME)/lib -L$(CU_HOME)/lib $(LDLIBS)
endif

# Compilation targets
.PHONY: all clean

all: $(PROGRAMS)

clean:
	-$(RM) *.o c63enc motion.o cuda_dct.o tables.o io.o c63_write.o me.o temp/test.c63


#c63enc: $(CLIENT_SRC)
#$(NVCC) -std=c++11 -x cu -ccbin $(CC) -Xcompiler $(INCLUDE) -o $@ $< -c -lm
#./c63enc -w 352 -h 288 -f 50 -r 4 -o temp/test.c63 temp/foreman.yuv

c63enc: $(CLIENT_OBJ)
	$(NVCC) $^ $(CFLAGS) $(LDFLAGS) -o $@ -lm

# How to compile CUDA
%.o: %.cu $(HEADERS)
	$(NVCC) -std=c++11 -x -g -G cu -ccbin $(CC) -Xcompiler "$(CFLAGS)" $(INCLUDE) -o $@ $< -c

# How to compile C++
%.o: %.cc $(HEADERS)
	$(CC) -x c++ -std=c++11 $(CFLAGS) $(INCLUDE) -o $@ $< -c

%.o: %.c  
	$(CC) $(CFLAGS) $(MFLAGS) $(addprefix .,$(patsubst %.o,%.d,$@)) -c $< -o $@
	
make foreman:
	./c63enc -f 300 -r 4 
	

make tractor:
	./c63enc -f 100 -r 4 
	
gprof:
	gprof c63enc gmon.out -b

-include $(CLIENT_DEPS) #$(SERVER_DEPS)
