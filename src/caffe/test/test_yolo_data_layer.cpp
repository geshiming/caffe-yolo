#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class YoloDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  
protected:
  YoloDataLayerTest()
      : seed_(1701),
	blob_top_data_(new Blob<Dtype>()),
	blob_top_label_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    // Size of cat is 480*360
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg ";
      outfile << "[" << i*70+35 << ", " << i*50+25 << ", " << 100 << ", " << 70 << "] " << 1 << std::endl;
    }
    outfile.close();
  }
  
  virtual ~YoloDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }
  
  int seed_;
  string filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(YoloDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(YoloDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  YoloDataParameter* yolo_data_param = param.mutable_yolo_data_param();
  yolo_data_param->set_batch_size(5);
  yolo_data_param->set_source(this->filename_.c_str());
  yolo_data_param->set_new_height(224);
  yolo_data_param->set_new_width(224);
  yolo_data_param->set_shuffle(false);
  yolo_data_param->set_num_sides(7);
  YoloDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 224);
  EXPECT_EQ(this->blob_top_data_->width(), 224);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 5);
  EXPECT_EQ(this->blob_top_label_->height(), 7);
  EXPECT_EQ(this->blob_top_label_->width(), 7);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < 5; i++) {
    int offset = this->blob_top_label_->offset(i, 0, i+1, i+1);
    float x = (i*70+85)/480.0;
    float y = (i*50+60)/360.0;
    float w = 100.0/480.0;
    float h = 70.0/360.0;

    Dtype* prefetch_label = this->blob_top_label_->mutable_cpu_data();
    EXPECT_EQ(prefetch_label[offset], 1);
    EXPECT_EQ(prefetch_label[offset+7*7], x);
    EXPECT_EQ(prefetch_label[offset+2*7*7], y);
    EXPECT_EQ(prefetch_label[offset+3*7*7], w);
    EXPECT_EQ(prefetch_label[offset+4*7*7], h);
  }
}

}
#endif	// USE_OPENCV