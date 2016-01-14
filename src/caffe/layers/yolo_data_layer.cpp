#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <math.h>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
caffe::YoloDataLayer<Dtype>::~YoloDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void caffe::YoloDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
						 const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.yolo_data_param().new_height();
  const int new_width = this->layer_param_.yolo_data_param().new_width();
  const bool is_color = this->layer_param_.yolo_data_param().is_color();
  string root_folder = this->layer_param_.yolo_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
  (new_height > 0 && new_width > 0)) << "Current implementation requires "
  "new_height and new_width to be set at the same time.";
  // Read the file with filenames and regions
  const string& source = this->layer_param_.yolo_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.is_open()) << "unable to find image data file";
  char file_line[4096];
  infile.getline(file_line, 4096);
  do {
    string str_line(file_line);
    string file_name = str_line.substr(0, str_line.find(" "));
    string file_truth_box = str_line.substr(str_line.find(" ") + 1, str_line.length());
    lines_.push_back(std::make_pair(file_name, file_truth_box));
    infile.getline(file_line, 4096);
  } while (!infile.eof());

  if (this->layer_param_.yolo_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.yolo_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() % this->layer_param_.yolo_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image to initialize the top blob
  // Here is a question for spp method: 
  // If image size is not defined, that is, crop size is not defined
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.yolo_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

  //label
  const int num_predictions = this->layer_param_.yolo_data_param().num_predictions();
  const int num_sides = this->layer_param_.yolo_data_param().num_sides();
  vector<int> label_shape(4);
  label_shape[0] = batch_size;
  label_shape[1] = 5;	//4 for coordinates and 1 for class label
  label_shape[2] = num_sides;
  label_shape[3] = num_sides;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void caffe::YoloDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void caffe::YoloDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  YoloDataParameter yolo_data_param = this->layer_param_.yolo_data_param();
  const int batch_size = yolo_data_param.batch_size();
  const int new_height = yolo_data_param.new_height();
  const int new_width = yolo_data_param.new_width();
  const bool is_color = yolo_data_param.is_color();
  string root_folder = yolo_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  const int num_predictions = this->layer_param_.yolo_data_param().num_predictions();
  const int num_sides = this->layer_param_.yolo_data_param().num_sides();
  vector<int> label_shape(4);
  label_shape[0] = batch_size;
  label_shape[1] = 5;	
  label_shape[2] = num_sides;
  label_shape[3] = num_sides;
  batch->label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  for (int i = 0; i < batch->data_.count(); i++) {
    prefetch_data[i] = 0;
  }
  for (int i = 0; i < batch->label_.count(); i++) {
    prefetch_label[i] = 0;
  }

  // datum scales
  // for each image, bounding box transformation is not implemented
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    vector<float* > 	box_coords;
    vector<int>		box_label;
    GetLabels(box_coords, box_label);

    //read image and transform box coordinates to relative value
    cv::Mat cv_img;
    ReadYoloImages(root_folder + lines_[lines_id_].first, new_height, new_width,
    is_color, cv_img, box_coords);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

    for (int box_id = 0; box_id < box_label.size(); box_id++) {
      //determine which grid is responsible for prediction 
      float *box_coord = box_coords[box_id];
      int label = box_label[box_id];
      int grid_x = floor(box_coord[0] * num_sides);
      int grid_y = floor(box_coord[1] * num_sides);
      
      // Set label to blob
      offset = batch->label_.offset(item_id, 0, grid_y, grid_x);
      prefetch_label[offset] = label;

      // Set truth box to blob
      for (int coord_id = 0; coord_id < 4; coord_id++) {
	offset = batch->label_.offset(item_id, coord_id+1, grid_y, grid_x);
	prefetch_label[offset] = box_coord[coord_id];
      }
    }

    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.yolo_data_param().shuffle()) {
	ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void caffe::YoloDataLayer<Dtype>::GetLabels(vector<float* >& box_coords, vector<int> &box_label) {
  std::string truth_box = lines_[lines_id_].second;
  while (truth_box.find(']') != -1) {
    std::string str_box = truth_box.substr(1, truth_box.find(']') - 1);
    float *box_coord = new float[4];
    for (int i = 0; i < 3; i++) {
      box_coord[i] = atof(str_box.substr(0, str_box.find(',')).c_str());
      str_box = str_box.substr(str_box.find(' ') + 1, str_box.length());
    }
    box_coord[3] = atof(str_box.c_str());
    box_coords.push_back(box_coord);

    truth_box = truth_box.substr(truth_box.find(']') + 2, truth_box.length());
    std::string str_label = truth_box.substr(0, truth_box.find(' '));
    box_label.push_back(atoi(str_label.c_str()));

    truth_box = truth_box.substr(truth_box.find(' ') + 1, truth_box.length());
  }
}

template <typename Dtype>
void caffe::YoloDataLayer<Dtype>::ReadYoloImages(const string& filename,
  const int height, const int width, const bool is_color, cv::Mat& cv_img, vector<float*>& truth_box) {
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);

  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return;
  }
  int width_origin = cv_img_origin.cols;
  int height_origin = cv_img_origin.rows;
  for (int i = 0; i < truth_box.size(); i++) {
    float *box_coord = truth_box[i];
    float x = box_coord[0];
    float y = box_coord[1];
    float w = box_coord[2];
    float h = box_coord[3];
    box_coord[0] = (x+w/2.0)/width_origin;
    box_coord[1] = (y+h/2.0)/height_origin;
    box_coord[2] = w/width_origin;
    box_coord[3] = h/height_origin;
  }

  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
}

INSTANTIATE_CLASS(YoloDataLayer);
REGISTER_LAYER_CLASS(YoloData);
}
#endif  // USE_OPENCV