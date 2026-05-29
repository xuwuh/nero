CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -Iinclude
LDFLAGS = -lm

SRC = c/main.c \
      c/config.c \
      c/load_dataset.c \
      c/matrix.c \
      c/utils.c \
      c/loss.c \
      c/optimizer.c \
      c/linear_layer.c \
      c/tensor.c \
      c/relu.c \
      c/maxpool.c \
      c/im2col.c \
      c/conv.c \
      c/flatten.c \
      c/model.c \
      c/train_cnn.c \
      c/test_cnn.c \
      c/log.c \
      c/noise.c \
      visualizations/model_save.c \
      visualizations/graph_save.c

OUT = nern.exe

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

run: all
	./$(OUT)

clean:
	rm -f $(OUT)
