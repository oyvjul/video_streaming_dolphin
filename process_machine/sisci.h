#include <inttypes.h>
#include "c63.h"

void init_sisci(int remote_node_id);
void init_sisci_dma();
void init_sisci_write(unsigned int local_adapter_no);
void init_local_encoded_data(uint32_t segment_size, unsigned int local_adapter_no, int remote_node_id);
void copy_segment(struct macroblock **mbs, dct_t* residuals);
void set_sizes_offsets(struct c63_common_cpu *cm);
void receive_width_height(int remote_node_id, unsigned int local_adapter_no, uint32_t *width, uint32_t *height);
struct yuv_segment_t receive_image_dma(int remote_node_id, unsigned int local_adapter_no, struct c63_common_cpu *cm);
struct yuv_segment_t receive_image_gpu_direct(int remote_node_id, unsigned int local_adapter_no, struct c63_common_gpu *cm);
void receive_width_height_pio(int remote_node_id, unsigned int local_adapter_no);
void wait_for_width_and_height(uint32_t *width, uint32_t *height);
void waiting_for_frame(int counter);
void ready_to_encode();
void done_encoding_frame(int counter);
void transfer_encoded_data();
void terminate_sisci_pio();
void terminate_sisci(unsigned int local_adapter_no);
