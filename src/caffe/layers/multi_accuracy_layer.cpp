#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"


namespace caffe {

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.multi_accuracy_param().has_ignore_label();
  ignore_label_ = -1;
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.multi_accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.multi_accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top){
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  Dtype accuracy = 0;
  int count = 0;
  for (int i = 0; i < outer_num_; i++) {
    for (int j = 0; j < inner_num_; j++) {
      bool valid = false;
      int truth_count = 0;
      for (int c = 0; c < num_labels; ++c) {
	if (label[i * dim + c * inner_num_ + j] == 1 && c != ignore_label_) {
	  truth_count++;
	}
      }
      for (int c = 0; c < num_labels; ++c) {
	if (label[i * dim + c * inner_num_ + j] == 1 && c != ignore_label_) {
	  accuracy += std::min(prob_data[i * dim + c * inner_num_ + j], Dtype(1.0 / truth_count));
	  valid = true;
	}
      }
      if (valid == true) {
	count++;
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = accuracy / count;
}

INSTANTIATE_CLASS(MultiAccuracyLayer);
REGISTER_LAYER_CLASS(MultiAccuracy);

}