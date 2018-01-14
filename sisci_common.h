#ifndef SISCI_COMMON_H
#define SISCI_COMMON_H

#include <stdint.h>

#define NO_FLAGS 0
#define NO_CALLBACK NULL
#define RECEIVER_INTERRUPT_NO 1234

#define GROUP 17

#ifndef GROUP
#error Fill in group number in common.h!
#endif

/* GET_SEGMENTIUD(2) gives you segmentid 2 at your groups offset */
#define GET_SEGMENTID(id) ( GROUP << 16 | id )

#define SEGMENT_READ_WRITE GET_SEGMENTID(4)
#define SEGMENT_PROCESSING GET_SEGMENTID(32)

#define PACKET_SIZE 64
#define DATA_SIZE   10

enum cmd
{
    CMD_INVALID,
    CMD_QUIT,
    CMD_ADD,
    CMD_DONE,
};

enum cmd_io
{
	SEND_FRAME,
	SEND_MORE,
	SENT_WIDTH_HEIGHT,
	READY_TO_SEND,
	WAITING
};

enum cmd_process
{
	RECEIVED,
	ENCODE_DONE,
	RECEIVE_MORE,
	RECEIVED_WIDTH_HEIGHT,
	READY_FOR_NEXT_FRAME
};

struct packet
{
    union
    {
        struct
        {
            uint32_t cmd;
            uint32_t param;
        };
    };
};


struct yuv_segment_t
{
	const volatile uint8_t *Y;
	const volatile uint8_t *U;
	const volatile uint8_t *V;
};

struct read_write_machine
{
	uint32_t cmd;
};

struct read_write_segment
{
    struct packet packet __attribute__((aligned(64)));
    uint32_t width_height[2];
};

struct processing_segment
{
    struct packet packet __attribute__((aligned(64)));
};

#endif
