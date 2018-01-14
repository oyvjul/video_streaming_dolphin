#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "gpu_data.cuh"
#include "c63_write.h"
#include "c63.h"
#include "tables.cuh"
#include "io.h"
#include "cuda_dct.h"
#include "motion.h"
#include "me.h"
extern "C"
{
#include "sisci.h"
}

FILE *outfile;
static int limit_numframes = 0;
static uint32_t width = 0;
static uint32_t height = 0;

/* getopt */
extern int optind;
extern char *optarg;

static void c63_encode_image_gpu(struct c63_common_cpu *cpu_cm, struct c63_common_gpu *gpu_cm, uint8_t* d_origY, uint8_t* d_origU, uint8_t* d_origV, uint8_t* d_ref_reconsY, uint8_t* d_ref_reconsU, uint8_t* d_ref_reconsV,
								uint8_t* d_current_reconsY, uint8_t* d_current_reconsU, uint8_t* d_current_reconsV, uint8_t* d_predictedY, uint8_t* d_predictedU, uint8_t* d_predictedV,
									struct macroblock *d_mbsY, struct macroblock *d_mbsU, struct macroblock *d_mbsV, int16_t *d_residualsYDCT, int16_t *d_residualsUDCT, int16_t *d_residualsVDCT)
{
	//zero out macroblock for every frame on the gpu
	cudaMemset((void*)d_mbsY, 0, gpu_cm->mb_rows * gpu_cm->mb_cols * sizeof(struct macroblock));
	cudaMemset((void*)d_mbsU, 0, gpu_cm->mb_rows/2 * gpu_cm->mb_cols/2 * sizeof(struct macroblock));
	cudaMemset((void*)d_mbsV, 0, gpu_cm->mb_rows/2 * gpu_cm->mb_cols/2 * sizeof(struct macroblock));

	//zero out macroblock for every frame on the cpu
	memset((void*)cpu_cm->mbs[Y_COMPONENT], 0, gpu_cm->mb_rows * gpu_cm->mb_cols * sizeof(struct macroblock));
	memset((void*)cpu_cm->mbs[U_COMPONENT], 0, gpu_cm->mb_rows/2 * gpu_cm->mb_cols/2 * sizeof(struct macroblock));
	memset((void*)cpu_cm->mbs[V_COMPONENT], 0, gpu_cm->mb_rows/2 * gpu_cm->mb_cols/2 * sizeof(struct macroblock));

	for(int i = 0; i < 3; ++i)
		  cudaStreamCreate(&streams[i]);


	// Check if keyframe
	if (gpu_cm->framenum == 0 || gpu_cm->frames_since_keyframe == gpu_cm->keyframe_interval)
	{
		cpu_cm->keyframe = 1;
		gpu_cm->frames_since_keyframe = 0;

		fprintf(stderr, " (keyframe) ");

		/*cudaMemcpyAsync(d_origY, (void*)cpu_cm->orig_segment.Y, gpu_cm->padw[Y_COMPONENT]*gpu_cm->padh[Y_COMPONENT], cudaMemcpyHostToDevice, streams[0]);
		cudaMemcpyAsync(d_origU, (void*)cpu_cm->orig_segment.U, gpu_cm->padw[U_COMPONENT]*gpu_cm->padh[U_COMPONENT], cudaMemcpyHostToDevice, streams[1]);
		cudaMemcpyAsync(d_origV, (void*)cpu_cm->orig_segment.V, gpu_cm->padw[V_COMPONENT]*gpu_cm->padh[V_COMPONENT], cudaMemcpyHostToDevice, streams[2]);*/

		cudaMemset((void*)d_predictedY, 0, gpu_cm->ypw * gpu_cm->yph * sizeof(uint8_t));
		cudaMemset((void*)d_predictedU, 0, gpu_cm->upw * gpu_cm->uph * sizeof(uint8_t));
		cudaMemset((void*)d_predictedV, 0, gpu_cm->vpw * gpu_cm->vph * sizeof(uint8_t));
	}
	else { cpu_cm->keyframe = 0; }

	if (!cpu_cm->keyframe)
	{
		//set the ref freame to current and send the current frame of the gpu, to the cpu
		d_ref_reconsY = d_current_reconsY;
		d_ref_reconsU = d_current_reconsU;
		d_ref_reconsV = d_current_reconsV;

		cudaMemcpyAsync((void*)cpu_cm->ref_recons_segment.Y, d_current_reconsY, gpu_cm->ypw * gpu_cm->yph, cudaMemcpyDeviceToHost, streams[0]);
		cudaMemcpyAsync((void*)cpu_cm->ref_recons_segment.U, d_current_reconsU, gpu_cm->upw * gpu_cm->uph, cudaMemcpyDeviceToHost, streams[1]);
		cudaMemcpyAsync((void*)cpu_cm->ref_recons_segment.V, d_current_reconsV, gpu_cm->vpw * gpu_cm->vph, cudaMemcpyDeviceToHost, streams[2]);

		/*cudaMemcpyAsync(d_origY, (void*)cpu_cm->orig_segment.Y, gpu_cm->padw[Y_COMPONENT]*gpu_cm->padh[Y_COMPONENT], cudaMemcpyHostToDevice, streams[0]);
		cudaMemcpyAsync(d_origU, (void*)cpu_cm->orig_segment.U, gpu_cm->padw[U_COMPONENT]*gpu_cm->padh[U_COMPONENT], cudaMemcpyHostToDevice, streams[1]);
		cudaMemcpyAsync(d_origV, (void*)cpu_cm->orig_segment.V, gpu_cm->padw[V_COMPONENT]*gpu_cm->padh[V_COMPONENT], cudaMemcpyHostToDevice, streams[2]);*/

		c63_motion_estimate_new(d_origY, d_origU, d_origV, d_ref_reconsY, d_ref_reconsU,
		d_ref_reconsV, d_mbsY, d_mbsU, d_mbsV, gpu_cm, streams);

		gpu_c63_motion_compensate(d_predictedY, d_predictedU, d_predictedV, d_ref_reconsY, d_ref_reconsU,
		d_ref_reconsV, d_mbsY, d_mbsU, d_mbsV, gpu_cm, streams);
	}

	dct_test(gpu_cm, d_origY, d_origU, d_origV, d_predictedY, d_predictedU, d_predictedV,
			d_residualsYDCT, d_residualsUDCT, d_residualsVDCT, streams,
			cpu_cm->residuals->Ydct, cpu_cm->residuals->Udct, cpu_cm->residuals->Vdct);

	idct_test(gpu_cm, d_predictedY, d_predictedU, d_predictedV, d_residualsYDCT,
			d_residualsUDCT, d_residualsVDCT, d_current_reconsY, d_current_reconsU,
			d_current_reconsV, streams);

	//macroblock can not be sent from device to host concurrently
	cudaMemcpyAsync(cpu_cm->mbs[Y_COMPONENT], d_mbsY, gpu_cm->mb_rows*gpu_cm->mb_cols*sizeof(struct macroblock), cudaMemcpyDeviceToHost, streams[0]);
	cudaMemcpyAsync(cpu_cm->mbs[U_COMPONENT], d_mbsU, gpu_cm->mb_rows/2 * gpu_cm->mb_cols/2 * sizeof(struct macroblock), cudaMemcpyDeviceToHost, streams[0]);
	cudaMemcpy(cpu_cm->mbs[V_COMPONENT], d_mbsV, gpu_cm->mb_rows/2 * gpu_cm->mb_cols/2 * sizeof(struct macroblock), cudaMemcpyDeviceToHost);

	++gpu_cm->framenum;
	++gpu_cm->frames_since_keyframe;
}

struct c63_common_gpu* init_c63_enc_gpu(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  struct c63_common_gpu *cm = (c63_common_gpu*) calloc(1, sizeof(struct c63_common_gpu));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  return cm;
}

struct c63_common_cpu* init_c63_enc_cpu(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  struct c63_common_cpu *cm = (c63_common_cpu*) calloc(1, sizeof(struct c63_common_cpu));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  cm->ref_recons = (yuv_t*) malloc(sizeof(yuv_t));
  cm->ref_recons->Y = (uint8_t*) malloc(cm->ypw * cm->yph);
  cm->ref_recons->U = (uint8_t*) malloc(cm->upw * cm->uph);
  cm->ref_recons->V = (uint8_t*) malloc(cm->vpw * cm->vph);

  cm->residuals = (dct_t*) malloc(sizeof(dct_t));
  cudaMallocHost((void**)&cm->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
  cudaMallocHost((void**)&cm->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
  cudaMallocHost((void**)&cm->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));

  cudaMallocHost((void**)&cm->mbs[Y_COMPONENT], cm->mb_rows * cm->mb_cols * sizeof(struct macroblock));
  cudaMallocHost((void**)&cm->mbs[U_COMPONENT], cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock));
  cudaMallocHost((void**)&cm->mbs[V_COMPONENT], cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock));

  return cm;
}

static void print_help()
{
  printf("%s\n", "Usage: ./c63enc [options] input_file");
  printf("%s\n", "Commandline options:");
  printf("%s\n", "  -h                             Height of images to compress");
  printf("%s\n", "  -w                             Width of images to compress");
  printf("%s\n", "  -o                             Output file (.c63)");
  printf("%s\n", "  [-f]                           Limit number of frames to encode");
  printf("%s\n", "\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;
  int remote_node_id;
  unsigned int local_adapter_no = 0;
  uint8_t *d_image_Y = 0;
  uint8_t *d_image_U = 0;
  uint8_t *d_image_V = 0;
  int16_t *d_residuals_YDCT = 0;
  int16_t *d_residuals_UDCT = 0;
  int16_t *d_residuals_VDCT = 0;
  uint8_t *d_curr_rec_Y = 0;
  uint8_t *d_curr_rec_U = 0;
  uint8_t *d_curr_rec_V = 0;
  uint8_t *d_ref_rec_Y = 0;
  uint8_t *d_ref_rec_U = 0;
  uint8_t *d_ref_reconsV = 0;
  uint8_t *d_predicted_Y = 0;
  uint8_t *d_predicted_U = 0;
  uint8_t *d_predicted_V = 0;
  struct macroblock *d_mbY = 0;
  struct macroblock *d_mbU = 0;
  struct macroblock *d_mbV = 0;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "f:r:")) != -1)
  {
    switch (c)
    {
      case 'f':
        limit_numframes = atoi(optarg);
        break;
      case 'r':
    	remote_node_id = atoi(optarg);
    	break;
      default:
        print_help();
        break;
    }
  }

  //SISCI
  struct yuv_segment_t array;
  init_sisci(remote_node_id);
  init_sisci_dma();
  init_sisci_write(local_adapter_no);
  receive_width_height_pio(remote_node_id, local_adapter_no);
  wait_for_width_and_height(&width, &height);

  struct c63_common_cpu *cpu_cm = init_c63_enc_cpu(width, height);
  struct c63_common_gpu *gpu_cm = init_c63_enc_gpu(width, height);

  uint32_t local_segment_size = sizeof(int);
  for (int c = 0; c < 3; ++c)
  {
	  local_segment_size += cpu_cm->mb_cols*cpu_cm->mb_rows*sizeof(struct macroblock);
	  local_segment_size += cpu_cm->padw[c]*cpu_cm->padh[c]*sizeof(int16_t);
  }

  set_sizes_offsets(cpu_cm);

  //array = receive_image_dma(remote_node_id, local_adapter_no, cpu_cm);
  array = receive_image_gpu_direct(remote_node_id, local_adapter_no, gpu_cm);
  init_local_encoded_data(local_segment_size, local_adapter_no, remote_node_id);

  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  clock_t start, end;
  double cpu_time_used;
  start = clock();

  int numframes = 0;

  //Allocate GPU Data
  cudaMalloc((void**)&d_ref_rec_Y, cpu_cm->ypw * cpu_cm->yph);
  cudaMalloc((void**)&d_ref_rec_U, cpu_cm->upw * cpu_cm->uph);
  cudaMalloc((void**)&d_ref_reconsV, cpu_cm->vpw * cpu_cm->vph);

  cudaMalloc((void**)&d_curr_rec_Y, cpu_cm->ypw * cpu_cm->yph);
  cudaMalloc((void**)&d_curr_rec_U, cpu_cm->upw * cpu_cm->uph);
  cudaMalloc((void**)&d_curr_rec_V, cpu_cm->vpw * cpu_cm->vph);

  cudaMalloc((void**)&d_mbY, cpu_cm->mb_rows * cpu_cm->mb_cols * sizeof(struct macroblock));
  cudaMalloc((void**)&d_mbU, cpu_cm->mb_rows/2 * cpu_cm->mb_cols/2 * sizeof(struct macroblock));
  cudaMalloc((void**)&d_mbV, cpu_cm->mb_rows/2 * cpu_cm->mb_cols/2 * sizeof(struct macroblock));

  /*cudaMalloc((void**)&d_image_Y, gpu_cm->padw[Y_COMPONENT]*gpu_cm->padh[Y_COMPONENT]);
  cudaMalloc((void**)&d_image_U, gpu_cm->padw[U_COMPONENT]*gpu_cm->padh[U_COMPONENT]);
  cudaMalloc((void**)&d_image_V, gpu_cm->padw[V_COMPONENT]*gpu_cm->padh[V_COMPONENT]);*/

  cudaMalloc((void**)&d_predicted_Y, cpu_cm->ypw * cpu_cm->yph);
  cudaMalloc((void**)&d_predicted_U, cpu_cm->upw * cpu_cm->uph);
  cudaMalloc((void**)&d_predicted_V, cpu_cm->vpw * cpu_cm->vph);

  cudaMalloc((void**)&d_residuals_YDCT, cpu_cm->ypw * cpu_cm->yph * sizeof(int16_t));
  cudaMalloc((void**)&d_residuals_UDCT, cpu_cm->upw * cpu_cm->uph * sizeof(int16_t));
  cudaMalloc((void**)&d_residuals_VDCT, cpu_cm->vpw * cpu_cm->vph * sizeof(int16_t));

  cudaMemset((void*)d_predicted_Y, 0, cpu_cm->ypw * cpu_cm->yph * sizeof(uint8_t));
  cudaMemset((void*)d_predicted_U, 0, cpu_cm->upw * cpu_cm->uph * sizeof(uint8_t));
  cudaMemset((void*)d_predicted_V, 0, cpu_cm->vpw * cpu_cm->vph * sizeof(uint8_t));

  cudaMemset((void*)d_residuals_YDCT, 0, cpu_cm->ypw * cpu_cm->yph * sizeof(int16_t));
  cudaMemset((void*)d_residuals_UDCT, 0, cpu_cm->upw * cpu_cm->uph * sizeof(int16_t));
  cudaMemset((void*)d_residuals_VDCT, 0, cpu_cm->vpw * cpu_cm->vph * sizeof(int16_t));

  while (1)
  {
	waiting_for_frame(numframes);
    //cpu_cm->orig_segment = array;

    printf("Encoding frame %d, ", numframes);
    /*c63_encode_image_gpu(cpu_cm, gpu_cm, (uint8_t*)array.Y, (uint8_t*)array.U, (uint8_t*)array.V, d_ref_rec_Y,
    			d_ref_rec_U, d_ref_reconsV, d_curr_rec_Y, d_curr_rec_U, d_curr_rec_V,
    			d_predicted_Y, d_predicted_U, d_predicted_V, d_mbY, d_mbU, d_mbV,
    			d_residuals_YDCT, d_residuals_UDCT, d_residuals_VDCT);*/
    d_image_Y = (uint8_t*)array.Y;
    d_image_U = (uint8_t*)array.U;
    d_image_V = (uint8_t*)array.V;

    c63_encode_image_gpu(cpu_cm, gpu_cm, d_image_Y, d_image_U, d_image_V, d_ref_rec_Y,
       			d_ref_rec_U, d_ref_reconsV, d_curr_rec_Y, d_curr_rec_U, d_curr_rec_V,
       			d_predicted_Y, d_predicted_U, d_predicted_V, d_mbY, d_mbU, d_mbV,
       			d_residuals_YDCT, d_residuals_UDCT, d_residuals_VDCT);

    copy_segment(cpu_cm->mbs, cpu_cm->residuals);
    transfer_encoded_data();

    printf("%s\n", "Done!");

    done_encoding_frame(numframes);

    ++numframes;
    if (limit_numframes && numframes >= limit_numframes) { break; }
  }

  cudaFree((void*)d_ref_rec_Y);
  cudaFree((void*)d_ref_rec_U);
  cudaFree((void*)d_ref_reconsV);

  cudaFree((void*)d_curr_rec_Y);
  cudaFree((void*)d_curr_rec_U);
  cudaFree((void*)d_curr_rec_V);

  cudaFree((void*)d_mbY);
  cudaFree((void*)d_mbU);
  cudaFree((void*)d_mbV);

  cudaFree((void*)d_image_Y);
  cudaFree((void*)d_image_U);
  cudaFree((void*)d_image_V);

  cudaFree((void*)d_predicted_Y);
  cudaFree((void*)d_predicted_U);
  cudaFree((void*)d_predicted_V);

  cudaFree((void*)d_residuals_YDCT);
  cudaFree((void*)d_residuals_UDCT);
  cudaFree((void*)d_residuals_VDCT);

  free(cpu_cm->ref_recons->Y);
  free(cpu_cm->ref_recons->U);
  free(cpu_cm->ref_recons->V);
  free(cpu_cm->ref_recons);

  cudaFreeHost((void*)cpu_cm->residuals->Ydct);
  cudaFreeHost((void*)cpu_cm->residuals->Udct);
  cudaFreeHost((void*)cpu_cm->residuals->Vdct);
  free(cpu_cm->residuals);

  cudaFreeHost((void*)cpu_cm->mbs[Y_COMPONENT]);
  cudaFreeHost((void*)cpu_cm->mbs[U_COMPONENT]);
  cudaFreeHost((void*)cpu_cm->mbs[V_COMPONENT]);

  int i;
  for(i = 0; i < 3; ++i)
  {
	  cudaStreamDestroy(streams[i]);
  }

  cudaDeviceReset();
  terminate_sisci_pio();

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Total time taken: %f \n", cpu_time_used);

  return EXIT_SUCCESS;
}

