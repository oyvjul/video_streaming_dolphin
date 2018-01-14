#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sisci_error.h>
#include <sisci_api.h>
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../sisci_common.h"
#include "sisci.h"
#include "c63.h"

sci_error_t 			error;
sci_desc_t              sd;
sci_desc_t              sd1;
sci_desc_t              sd2;
sci_remote_segment_t    remote_segment;
sci_local_segment_t     local_segment;
sci_remote_segment_t    remote_segment1;
sci_remote_segment_t    remote_segment2;
sci_local_segment_t     local_segment1;
sci_local_segment_t     local_segment2;
sci_map_t               local_map;
sci_map_t               remote_map;
sci_map_t               local_map1;
sci_map_t               local_map2;
sci_map_t               remote_map1;
sci_map_t               remote_map2;
sci_dma_queue_t dma_queue;
sci_dma_queue_state_t dma_queue_state;

volatile struct processing_segment *processing_segment;
volatile struct read_write_segment *read_write_segment;
unsigned int segment_size = 0;
unsigned int segment_size_Y = 0;
unsigned int segment_size_U = 0;
unsigned int segment_size_V = 0;
unsigned int segment_size_writer = 0;
unsigned int keyframe_size;
unsigned int mb_size;
unsigned int residuals_sizes[COLOR_COMPONENTS];
unsigned int keyframe_offset;
unsigned int mb_offsets[COLOR_COMPONENTS];
unsigned int residuals_offsets[COLOR_COMPONENTS];
int *keyframe;
struct macroblock *mb_Y;
struct macroblock *mb_U;
struct macroblock *mb_V;
dct_t *residuals_Y;
dct_t *residuals_U;
dct_t *residuals_V;
volatile uint8_t* cuda_buffers;

void init_sisci(int remote_node_id)
{
	SCIInitialize(NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCIInitialize failed - Error code: 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	SCIOpen(&sd,NO_FLAGS,&error);
	if (error != SCI_ERR_OK)
	{
		if (error == SCI_ERR_INCONSISTENT_VERSIONS)
			fprintf(stderr,"Version mismatch between SISCI user library and SISCI driver\n");
		fprintf(stderr,"SCIOpen failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}
}

void init_sisci_dma()
{
	SCIOpen(&sd1,NO_FLAGS,&error);
	if (error != SCI_ERR_OK)
	{
		if (error == SCI_ERR_INCONSISTENT_VERSIONS)
			fprintf(stderr,"Version mismatch between SISCI user library and SISCI driver\n");
		fprintf(stderr,"SCIOpen failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}
}

void init_sisci_write(unsigned int local_adapter_no)
{
	SCIOpen(&sd2, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		if (error == SCI_ERR_INCONSISTENT_VERSIONS)
			fprintf(stderr,"Version mismatch between SISCI user library and SISCI driver\n");
		fprintf(stderr,"SCIOpen failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	SCICreateDMAQueue(sd2, &dma_queue, local_adapter_no, 1, NO_FLAGS, &error);
	if(error != SCI_ERR_OK)
	{
		fprintf(stderr,"CreateDMAQueue failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}
}

void set_sizes_offsets(struct c63_common_cpu *cm)
{
	static const int Y = Y_COMPONENT;
	static const int U = U_COMPONENT;
	static const int V = V_COMPONENT;

    keyframe_size = sizeof(int);
    mb_size = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
    residuals_sizes[Y] = cm->ypw * cm->yph * sizeof(int16_t);
    residuals_sizes[U] = cm->upw * cm->uph * sizeof(int16_t);
    residuals_sizes[V] = cm->vpw * cm->vph * sizeof(int16_t);

    keyframe_offset = 0;
    mb_offsets[Y] = keyframe_offset + keyframe_size;
    mb_offsets[U] = mb_offsets[Y] + mb_size;
    mb_offsets[V] = mb_offsets[U] + mb_size;
    residuals_offsets[Y] = mb_offsets[V] + mb_size;
    residuals_offsets[U] = residuals_offsets[Y] + residuals_sizes[Y];
    residuals_offsets[V] = residuals_offsets[U] + residuals_sizes[U];

}

void init_local_encoded_data(uint32_t segment_size_local, unsigned int local_adapter_no, int remote_node_id)
{
	segment_size_writer = segment_size_local;

	SCICreateSegment(sd2, &local_segment2, SEGMENT_PROCESSING+2, segment_size_writer,
			NO_CALLBACK, NULL, NO_FLAGS, &error);
	if(error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCICreateSegment dma failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	SCIPrepareSegment(local_segment2,local_adapter_no, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCIPrepareSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	uint8_t *buffer = SCIMapLocalSegment(local_segment2, &local_map2, 0, segment_size_writer,
			NULL, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCIMapLocalSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	keyframe = (int*) ((uint8_t*)buffer + keyframe_offset);

	mb_Y = (struct macroblock*) ((uint8_t*) buffer + mb_offsets[Y_COMPONENT]);
	mb_U = (struct macroblock*) ((uint8_t*) buffer + mb_offsets[U_COMPONENT]);
	mb_V = (struct macroblock*) ((uint8_t*) buffer + mb_offsets[V_COMPONENT]);

	residuals_Y = (dct_t*) ((uint8_t*) buffer + residuals_offsets[Y_COMPONENT]);
	residuals_U = (dct_t*) ((uint8_t*) buffer + residuals_offsets[U_COMPONENT]);
	residuals_V = (dct_t*) ((uint8_t*) buffer + residuals_offsets[V_COMPONENT]);

	printf("Connecting to writer...\n");
	do
	{
		SCIConnectSegment(sd2, &remote_segment2, remote_node_id, SEGMENT_READ_WRITE+2, local_adapter_no,
				NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);
	}
	while(error != SCI_ERR_OK);
	printf("Connected segment succsesfully to writer!\n");
}

void copy_segment(struct macroblock **mbs, dct_t* residuals)
{
	memcpy(mb_Y, mbs[Y_COMPONENT], mb_size);
	memcpy(mb_U, mbs[U_COMPONENT], mb_size);
	memcpy(mb_V, mbs[V_COMPONENT], mb_size);

	memcpy(residuals_Y, residuals->Ydct, residuals_sizes[Y_COMPONENT]);
	memcpy(residuals_U, residuals->Udct, residuals_sizes[U_COMPONENT]);
	memcpy(residuals_V, residuals->Vdct, residuals_sizes[V_COMPONENT]);
}

void receive_width_height(int remote_node_id, unsigned int local_adapter_no, uint32_t *width, uint32_t *height)
{
	printf("debug");
}

struct yuv_segment_t receive_image_dma(int remote_node_id, unsigned int local_adapter_no, struct c63_common_cpu *cm)
{
	struct yuv_segment_t array;

	segment_size_Y = cm->ypw*cm->yph*sizeof(uint8_t);
	segment_size_U = cm->upw*cm->uph*sizeof(uint8_t);
	segment_size_V = cm->vpw*cm->vph*sizeof(uint8_t);
	segment_size = segment_size_Y + segment_size_U + segment_size_V;

	SCICreateSegment(sd1, &local_segment1, SEGMENT_PROCESSING+1, segment_size,
			NO_CALLBACK, NULL, NO_FLAGS, &error);
	if(error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCICreateSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	SCIPrepareSegment(local_segment1,local_adapter_no, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCIPrepareSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	uint8_t *buffer = SCIMapLocalSegment(local_segment1, &local_map1, 0, segment_size,
			NULL, NO_FLAGS, &error);

	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCIMapLocalSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	SCISetSegmentAvailable(local_segment1, local_adapter_no, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCISetSegmentAvailable failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	unsigned int offset = 0;
	array.Y = buffer;
	array.Y = buffer + offset;
	offset += segment_size_Y;
	array.U = buffer + offset;
	offset += segment_size_U;
	array.V = buffer + offset;
	offset += segment_size_V;

	return array;
}

struct yuv_segment_t receive_image_gpu_direct(int remote_node_id, unsigned int local_adapter_no, struct c63_common_gpu *cm)
{
	struct yuv_segment_t array;

	segment_size_Y = cm->ypw*cm->yph*sizeof(uint8_t);
	segment_size_U = cm->upw*cm->uph*sizeof(uint8_t);
	segment_size_V = cm->vpw*cm->vph*sizeof(uint8_t);
	segment_size = segment_size_Y + segment_size_U + segment_size_V;

	SCICreateSegment(sd1, &local_segment1, SEGMENT_PROCESSING+1, segment_size,
			NO_CALLBACK, NULL, SCI_FLAG_EMPTY, &error);
	if(error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCICreateSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	cudaMalloc((void*)&cuda_buffers, segment_size);

	SCIAttachPhysicalMemory(0, (void*)cuda_buffers, 0, segment_size, local_segment1, SCI_FLAG_CUDA_BUFFER, &error);
	if(error != SCI_ERR_OK)
	{
		fprintf(stderr,"ATTACH FAILED - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	uint8_t *buffer = SCIMapLocalSegment(local_segment1, &local_map1, 0, segment_size,
				NULL, NO_FLAGS, &error);

	unsigned int offset = 0;
	array.Y = buffer;
	array.Y = buffer + offset;
	offset += segment_size_Y;
	array.U = buffer + offset;
	offset += segment_size_U;
	array.V = buffer + offset;
	offset += segment_size_V;

	SCIPrepareSegment(local_segment1,local_adapter_no, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCIPrepareSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	SCISetSegmentAvailable(local_segment1, local_adapter_no, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCISetSegmentAvailable failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	return array;
}

void receive_width_height_pio(int remote_node_id, unsigned int local_adapter_no)
{
	volatile struct image_common2 *common;
	struct yuv_segment_t array;

	SCICreateSegment(sd, &local_segment, SEGMENT_PROCESSING, sizeof(struct processing_segment),
			NO_CALLBACK, NULL, NO_FLAGS, &error);
	if(error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCICreateSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	SCIPrepareSegment(local_segment,local_adapter_no, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCIPrepareSegment failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

	SCISetSegmentAvailable(local_segment, local_adapter_no, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
	{
		fprintf(stderr,"SCISetSegmentAvailable failed - Error code 0x%x\n",error);
		exit(EXIT_FAILURE);
	}

    printf("Connecting...\n");
    do
    {
    	SCIConnectSegment(sd, &remote_segment, remote_node_id, SEGMENT_READ_WRITE,
    			local_adapter_no, NO_CALLBACK, NULL, SCI_INFINITE_TIMEOUT, NO_FLAGS, &error);

    }while(error != SCI_ERR_OK);
    printf("Connected segment succsesfully!\n");

    processing_segment = SCIMapLocalSegment(local_segment, &local_map, 0, sizeof(struct processing_segment),
    		NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK)
    {
    	fprintf(stderr,"SCIMapLocalSegment failed - Error code 0x%x\n",error);
    	exit(EXIT_FAILURE);
    }

    read_write_segment = SCIMapRemoteSegment(remote_segment, &remote_map, 0, sizeof(struct read_write_segment),
    		NULL, NO_FLAGS, &error);
    if (error != SCI_ERR_OK)
    {
    	fprintf(stderr,"SCIMapRemoteSegment failed - Error code 0x%x\n",error);
    	exit(EXIT_FAILURE);
    }
}

void wait_for_width_and_height(uint32_t *width, uint32_t *height)
{
	*width = 0;
	*height = 0;

	while(processing_segment->packet.cmd != SENT_WIDTH_HEIGHT);

	*width = read_write_segment->width_height[0];
	*height = read_write_segment->width_height[1];

	read_write_segment->packet.cmd = RECEIVED_WIDTH_HEIGHT;

	printf("Done receiving width: %d & height: %d \n", *width, *height);
}

void waiting_for_frame(int counter)
{
	while(processing_segment->packet.cmd != READY_TO_SEND+counter);
	printf("ready to receive new frame... \n");
}

void ready_to_encode()
{
	while(read_write_segment->packet.cmd != READY_TO_SEND);
	printf("Start encoding...\n");
}

void done_encoding_frame(int counter)
{
	read_write_segment->packet.cmd = READY_FOR_NEXT_FRAME+counter;
	printf("done encoding frame... \n");
}

void transfer_encoded_data()
{
	SCIStartDmaTransfer(dma_queue, local_segment2, remote_segment2, 0, segment_size_writer,
			0, NO_CALLBACK, NULL, NO_FLAGS, &error);
}

void terminate_sisci_pio()
{
	SCITerminate();
}

void terminate_sisci(unsigned int local_adapter_no)
{
	printf("terminate1...\n");
	SCISetSegmentUnavailable(local_segment, local_adapter_no, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
			fprintf(stderr,"SetSegmentUnavailable went wrong\n");

	printf("terminate2...\n");
	SCIUnmapSegment(local_map, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
		fprintf(stderr,"unmapping went wrong\n");

	printf("terminate3...\n");
	SCIRemoveSegment(local_segment, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
		fprintf(stderr,"remove segment went wrong\n");

	printf("terminate4...\n");
	SCIClose(sd, NO_FLAGS, &error);
	if (error != SCI_ERR_OK)
		fprintf(stderr,"SCI close went wrong\n");

	printf("terminate5...\n");
	SCITerminate();
}
