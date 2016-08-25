#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include <ecto_image_pipeline/pinhole_camera_model.h>
#include <ecto_image_pipeline/enums.hpp>

using ecto::tendrils;

namespace image_pipeline
{
  struct CvRectifier
  {
    static void
    declare_params(tendrils& params)
    {
      params.declare<InterpolationMode>("interpolation_mode", "Interpolation method for rectification", CV_INTER_LINEAR);
      params.declare<double>("cx_offset", "Center offset X of input image", 0.0);
      params.declare<double>("cy_offset", "Center offset Y of input image", 0.0);
    }

    static void
    declare_io(const tendrils& params, tendrils& in, tendrils& out)
    {
      in.declare(&CvRectifier::K_, "K").required(true);
      in.declare(&CvRectifier::D_, "D").required(true);
      in.declare(&CvRectifier::image_size_,"image_size").required(true);
      in.declare(&CvRectifier::image_in_, "image_in", "The input image.").required(true);
      out.declare(&CvRectifier::image_out_,"image_out", "The rectified image.");
    }

    void
    configure(const tendrils& params, const tendrils& in, const tendrils& out)
    {
      params["interpolation_mode"] >> mode_;
      params["cx_offset"] >> cx_offset_;
      params["cy_offset"] >> cy_offset_;
    }

    int
    process(const tendrils&, const tendrils&)
    {
      PinholeCameraModel new_model_;
      new_model_.setParams(*image_size_, *K_, *D_, cv::Mat(), cv::Mat(), cx_offset_, cy_offset_);

      if(new_model_ != cam_model_)
      {
        cam_model_ = new_model_;
        cam_model_.initCache();
      }

      cv::Mat output;
      cam_model_.rectifyImage(*image_in_, output, mode_);
      *image_out_ = output;
      return ecto::OK;
    }

    ecto::spore<cv::Mat> K_, D_, P_;
    ecto::spore<cv::Size> image_size_;
    ecto::spore<cv::Mat> image_in_, image_out_;
    PinholeCameraModel cam_model_;
    InterpolationMode mode_;
    double cx_offset_, cy_offset_;
  };
}

ECTO_CELL(base, image_pipeline::CvRectifier, "CvRectifier",
          "Given a opencv style camera info, rectify the input image.");
