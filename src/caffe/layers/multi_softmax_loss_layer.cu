#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultiSoftmaxLossForwardGPU(const int nthreads,
	  const Dtype* prob_data, const Dtype* label, Dtype* loss,
	  const int num, const int dim, const int spatial_dim,
	  const bool has_ignore_label_, const int ignore_label_, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_num = dim / spatial_dim;
    for (int c = 0; c < label_num; ++c) {
      int label_index = n * dim + c * spatial_dim + s;
      if (label[label_index] == 1 && c != ignore_label_) {
	loss[label_index] = -log(max(prob_data[label_index], Dtype(FLT_MIN)));
	counts[label_index] = 1;
      } else {
	loss[label_index] = 0;
	counts[label_index] = 0;
      }
    }
  }
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Forward_gpu (
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* counts = prob_.mutable_gpu_diff();
  MultiSoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(prob_.count(), loss_data, &loss);
  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(prob_.count(), counts, &count);
    loss /= count;
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void MultiSoftmaxLossBackwardGPU(const int nthreads, const Dtype* label,
	  Dtype* bottom_diff, const int num, const int dim, const int spatial_dim,
	  const bool has_ignore_label_, const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int truth_count = 0;
    for (int c = 0; c < channels; ++c) {
      if (label[n * dim + c * spatial_dim + s] == 1 && c != ignore_label_) {
	truth_count++;
      }
    }

    for (int c = 0; c < channels; ++c) {
      int label_index = n * dim + c * spatial_dim + s;
      bottom_diff[label_index] *= truth_count;
      if (label[label_index] == 1 && c != ignore_label_) {
	bottom_diff[label_index] -= 1;
	counts[label_index] = 1;
      }
    }

  }
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Backward_gpu (const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs. ";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    Dtype* counts = prob_.mutable_gpu_diff();
    MultiSoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, bottom_diff, outer_num_,
	dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(prob_.count(), counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiSoftmaxWithLossLayer);

}