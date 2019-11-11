#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>
#include <cassert>

// Graphs
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Pose3.h>
// Factors
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include "graph_optimizer/PoseGraph.hpp"
#include "utils/data_type.hpp"
#include "utils/conversions.hpp"
namespace cvo {

  using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

  static std::string traj_file_name = "full_traj.txt";
  
  PoseGraph::PoseGraph(bool is_f2f, bool use_sliding_window):
    isam2_(nullptr),
    is_f2f_(is_f2f),
    using_sliding_window_(use_sliding_window){
    // ISAM2 solver
    gtsam::ISAM2Params isam_params;
    isam_params.relinearizeThreshold = 0.01;
    isam_params.relinearizeSkip = 1;
    isam_params.cacheLinearizedFactors = false;
    isam_params.enableDetailedResults = true;
    isam_params.print();
    this->isam2_ .reset( new gtsam::ISAM2(isam_params));

    std::ofstream outfile;
    outfile.open(traj_file_name, std::ofstream::out| std::ofstream::trunc  );
    outfile.close();

    all_frames_since_last_keyframe_ = {};
    
  }

  PoseGraph::~PoseGraph() {
    
    
  }

  static Eigen::Affine3f read_tracking_init_guess() {
    FILE * tracking_init_guess_f = fopen("cvo_init.txt", "r");
    Eigen::Affine3f init_guess = Eigen::Affine3f::Identity();
    auto & m = init_guess.matrix();
    if (tracking_init_guess_f) {
      fscanf(tracking_init_guess_f,
             "%f %f %f %f %f %f %f %f %f %f %f %f\n",
             &m(0,0), &m(0,1), &m(0,2), &m(0,3),
             &m(1,0), &m(1,1), &m(1,2), &m(1,3),
             &m(2,0), &m(2,1), &m(2,2), &m(2,3));
      fclose(tracking_init_guess_f);
    } else {
      printf("No tracking init guess file found! use identity\n");
      m.setIdentity();
    }
    printf("First init guess of cvo tracking is \n");
    std::cout<<m<<std::endl;
    return init_guess;
  }

  Eigen::Affine3f PoseGraph::compute_frame_pose_in_graph(std::shared_ptr<Frame> frame)  {
    Eigen::Affine3f output;
    if (frame->is_keyframe()) {
      output = frame->pose_in_graph();
    } else {
      int ref_id = frame->tracking_pose_from_last_keyframe().ref_frame_id();
      Eigen::Affine3f ref_frame_pose = id2keyframe_[ref_id]->pose_in_graph();
      output = ref_frame_pose * frame->tracking_pose_from_last_keyframe().ref_frame_to_curr_frame();
    }
    return output;
  }

  bool PoseGraph::decide_new_keyframe(std::shared_ptr<Frame> new_frame,
                                      const Aff3f & pose_from_last_keyframe,
                                      float & inner_product_from_last_keyframe) {
    if (tracking_relative_transforms_.size() == 0)
      return true;
    
    bool is_keyframe = true;
    auto last_kf = all_frames_since_last_keyframe_[0];
    inner_product_from_last_keyframe = cvo_align_.inner_product(last_kf->points(), new_frame->points(), pose_from_last_keyframe);
    return is_tracking_bad(inner_product_from_last_keyframe);
  }

  bool PoseGraph::is_tracking_bad(float inner_product) const {
    return inner_product < 0.21;
    
  }
  
  Eigen::Affine3f PoseGraph::compute_tracking_pose_from_last_keyframe(const Eigen::Affine3f & tracking_pose_from_last_frame,
                                                                      std::shared_ptr<Frame> tracking_ref )  {
    Eigen::Affine3f keyframe_to_new_pose;
    if (is_f2f_) {

      if (!tracking_ref->is_keyframe()) {
        auto last_kf_id = tracking_ref->tracking_pose_from_last_keyframe().ref_frame_id(); 
        auto last_kf_to_tracking_ref = tracking_ref->tracking_pose_from_last_keyframe().ref_frame_to_curr_frame();
        // tracking_ref is not a keyframe
        keyframe_to_new_pose = last_kf_to_tracking_ref * tracking_pose_from_last_frame;
      } else
        // tracking_ref itself is the keyframes
        keyframe_to_new_pose = tracking_pose_from_last_frame;
      return keyframe_to_new_pose;
    } else {
      return tracking_pose_from_last_frame;
    }
  }

  RelativePose PoseGraph::track_from_last_frame(std::shared_ptr<Frame> new_frame) {

    RelativePose tracking_pose(new_frame->id);
    Eigen::Affine3f cvo_init = Aff3f::Identity();
    
    auto  last_frame = last_two_frames_.back();
    auto  slast_frame = last_two_frames_.front();

    Eigen::Affine3f slast_frame_pose_in_graph = compute_frame_pose_in_graph(slast_frame);
    Eigen::Affine3f last_frame_pose_in_graph = compute_frame_pose_in_graph(last_frame);
    Eigen::Affine3f slast_frame_to_last_frame = slast_frame_pose_in_graph.inverse() * last_frame_pose_in_graph;
    

    auto & curr_points = new_frame->points();
    int ref_frame_id = last_frame->id;
    auto & last_frame_points = last_frame->points();

    if (keyframes_.size())
      cvo_init = slast_frame_to_last_frame.inverse();
    else
      cvo_init = read_tracking_init_guess();

    printf("Call cvo.align from frame %d to frame %d\n", ref_frame_id, new_frame->id);
    cvo_align_.set_pcd(last_frame_points, curr_points, cvo_init, true);

    int align_ret = cvo_align_.align();
    Aff3f track_result = cvo_align_.get_transform();
    float inner_prod = cvo_align_.inner_product();
    std::cout<<"Cvo Align Result between "<<ref_frame_id<<" and "<<new_frame->id
             <<",inner product "<<cvo_align_.inner_product() <<", transformation is \n" <<track_result.matrix()<<"\n";
      
    if (align_ret == 0)
      tracking_pose.set_relative_transform(last_frame->id, track_result, inner_prod);
    else {
      tracking_pose.set_relative_transform(last_frame->id, track_result, 0.0);
    }
    return tracking_pose;
  }

  RelativePose PoseGraph::track_from_last_keyframe(std::shared_ptr<Frame> new_frame) {
    auto  last_keyframe = all_frames_since_last_keyframe_[0];
    auto  last_frame = last_two_frames_.back();
    auto slast_frame = last_two_frames_.front();
    Aff3f cvo_init = Aff3f::Identity();
    Eigen::Affine3f last_frame_pose_in_graph = compute_frame_pose_in_graph(last_frame);
    Eigen::Affine3f last_keyframe_to_last_frame = last_keyframe->pose_in_graph().inverse() * last_frame_pose_in_graph;
    Eigen::Affine3f slast_frame_pose_in_graph = compute_frame_pose_in_graph(slast_frame);
    Eigen::Affine3f slast_frame_to_last_frame = slast_frame_pose_in_graph.inverse() * last_frame_pose_in_graph;

    auto & curr_points = new_frame->points();
    int ref_frame_id = last_keyframe->id;
    auto & last_kf_points = last_keyframe->points();
    if (keyframes_.size()) 
      cvo_init = (last_keyframe_to_last_frame * slast_frame_to_last_frame).inverse();
    else
      cvo_init = read_tracking_init_guess();
    printf("Call cvo.align from frame %d to frame %d\n", ref_frame_id, new_frame->id);
    cvo_align_.set_pcd(last_kf_points, curr_points, cvo_init, true);
    
    int align_ret = cvo_align_.align();
    Aff3f track_result = cvo_align_.get_transform();
    float inner_prod = cvo_align_.inner_product();
    std::cout<<"Cvo Align Result between "<<ref_frame_id<<" and "<<new_frame->id
             <<",inner product "<<cvo_align_.inner_product() <<", transformation is \n" <<track_result.matrix()<<"\n";

    RelativePose tracking_pose(new_frame->id);
    if (align_ret == 0)
      tracking_pose.set_relative_transform(last_keyframe->id, track_result, inner_prod);
    else {
      tracking_pose.set_relative_transform(last_keyframe->id, track_result, 0.0);
    }
    return tracking_pose;

  }

  RelativePose tracking_from_last_keyframe_map(std::shared_ptr<Frame> new_frame)  {
    std::cerr<<"Not implement Error"<<std::endl;
    assert(0);
  }
  

  float PoseGraph::track_new_frame(std::shared_ptr<Frame> new_frame,
                                   bool & is_keyframe) {

    if (tracking_relative_transforms_.size() == 0) {
      is_keyframe = true;
      new_frame->set_relative_transform_from_ref(new_frame->id, Aff3f::Identity() , 1); // set to the frame itself

    } else {
      // at least one frame
      RelativePose tracking_result(new_frame->id);
      Eigen::Affine3f pose_from_last_kf;
      if (is_f2f_) {
        tracking_result = track_from_last_frame(new_frame);
        pose_from_last_kf = compute_tracking_pose_from_last_keyframe(tracking_result.ref_frame_to_curr_frame(), last_two_frames_.back()  );      
      } else {
        tracking_result = track_from_last_keyframe(new_frame);
        pose_from_last_kf = tracking_result.ref_frame_to_curr_frame();
      }

      float inner_product_from_last_kf;
      is_keyframe = decide_new_keyframe(new_frame, pose_from_last_kf, inner_product_from_last_kf);
      printf("the new frame %d 's is_keframe inner product from last kf %d is %f\n", new_frame->id, all_frames_since_last_keyframe_[0]->id, inner_product_from_last_kf);

      if (is_f2f_) {
        if (is_keyframe)
          new_frame->set_relative_transform_from_ref(tracking_result);
        else
          new_frame->set_relative_transform_from_ref(all_frames_since_last_keyframe_[0]->id, pose_from_last_kf, inner_product_from_last_kf );
      } else  {
        if (is_keyframe) {
          RelativePose last_frame_to_new_frame_pose = track_from_last_frame(new_frame);
          new_frame->set_relative_transform_from_ref(last_frame_to_new_frame_pose );
        } else
          new_frame->set_relative_transform_from_ref(tracking_result);
      } 
    }
    tracking_relative_transforms_.push_back(new_frame->tracking_pose_from_last_keyframe());
    new_frame->set_keyframe(is_keyframe);
    return new_frame->tracking_pose_from_last_keyframe().cvo_inner_product();
  }
  
  void PoseGraph::add_new_frame(std::shared_ptr<Frame> new_frame) {
    std::cout<<"add_new_frame: id "<<new_frame->id<<std::endl;
    std::cout<<"---- number of points is "<<new_frame->points().num_points()<<std::endl;
    //new_frame->points().write_to_color_pcd(std::to_string(new_frame->id)+".pcd"  );
    bool is_keyframe = false;

    // tracking
    track_new_frame(new_frame, is_keyframe);

    // deal with keyframe and nonkeyframe
    printf("Tracking: is keyframe is %d\n", is_keyframe);
    if(is_keyframe) {
      // pose graph optimization
      id2keyframe_[new_frame->id] = new_frame;
      if (keyframes_.size() == 0) {
        init_pose_graph(new_frame);
      } else {
        pose_graph_optimize(new_frame);
      }
      keyframes_.push_back (new_frame);
      all_frames_since_last_keyframe_.clear();
      new_frame->construct_map();
      
    } else {
      int ref_frame_id = tracking_relative_transforms_[new_frame->id].ref_frame_id();
      auto ref_frame = id2keyframe_[ref_frame_id];
      ref_frame->add_points_to_map_from(*new_frame);
    }

    // maintain the data structures in PoseGraph.hpp
    all_frames_since_last_keyframe_.push_back(new_frame);
    last_two_frames_.push(new_frame);
    if (last_two_frames_.size() > 2)
      last_two_frames_.pop();
    
    new_frame->set_keyframe(is_keyframe);

    static uint32_t counter = 0;
    if (is_keyframe){
      if (counter % 3 == 0)
        write_trajectory(traj_file_name );
      counter++;
    }
  }

  void PoseGraph::init_pose_graph(std::shared_ptr<Frame> new_frame) {
    //fill config values
    gtsam::Vector4 q_WtoC;
    q_WtoC << 0,0,0,1;
    gtsam::Vector3 t_WtoC;
    t_WtoC << 0,0,0;
    gtsam::Vector6 prior_pose_noise;
    prior_pose_noise << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.1);

    // prior state and noise
    gtsam::Pose3 prior_state(gtsam::Quaternion(q_WtoC(3), q_WtoC(0), q_WtoC(1), q_WtoC(2)),
                             t_WtoC);
    auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas( prior_pose_noise);
    
    //factor_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(new_frame->id),
    //                                                   prior_state, pose_noise));
    factor_graph_.add(gtsam::PriorFactor<gtsam::Pose3>((new_frame->id),
                                                       prior_state, pose_noise));
    //graph_values_.insert(X(new_frame->id), prior_state);
    graph_values_.insert((new_frame->id), prior_state);
    //key2id_[X(new_frame->id)] = id;

    factor_graph_.print("gtsam Initial Graph\n");
    
  }

  void PoseGraph::pose_graph_optimize(std::shared_ptr<Frame> new_frame) {

    assert(tracking_relative_transforms_.size() > 1);
    
    int new_id = new_frame->id;
    int last_kf_id = all_frames_since_last_keyframe_[0]->id;
    auto last_kf = id2keyframe_[last_kf_id];
    auto last_frame = last_two_frames_.back();
    Eigen::Affine3f tf_last_keyframe_to_last_frame;
    if (last_kf->id == last_frame->id ) 
      tf_last_keyframe_to_last_frame = Eigen::Affine3f::Identity();
    else
      tf_last_keyframe_to_last_frame = tracking_relative_transforms_[last_frame->id].ref_frame_to_curr_frame();
    auto tf_last_keyframe_to_newframe = tf_last_keyframe_to_last_frame * tracking_relative_transforms_[new_id].ref_frame_to_curr_frame();

    Eigen::Affine3f tf_WtoNew_eigen = last_kf->pose_in_graph() * tf_last_keyframe_to_newframe;
    gtsam::Pose3 tf_WtoNew = affine3f_to_pose3(tf_WtoNew_eigen);
    gtsam::Pose3 odom_last_kf_to_new = affine3f_to_pose3(tf_last_keyframe_to_newframe);
    // TODO? use the noise from inner product??
    gtsam::Vector6 prior_pose_noise;
    prior_pose_noise << gtsam::Vector3::Constant(0.1), gtsam::Vector3::Constant(0.1);
    auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas( prior_pose_noise);
    std::cout<<"optimize the pose graph with gtsam...\n";
    std::cout<<" new frames's tf_WtoNew "<<tf_WtoNew;
    std::cout<<" new frames' odom_last_kf_to_new"<<odom_last_kf_to_new<<"";
    // factor_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(last_kf_id), X(new_id),
    //                                                    odom_last_kf_to_new, pose_noise));
    factor_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>((last_kf_id), (new_id),
                                                         odom_last_kf_to_new, pose_noise));
    // TOOD: add init value for this state
    //graph_values_.insert(X(new_id), tf_WtoNew);
    graph_values_.insert((new_id), tf_WtoNew);
    //key2id_
    std::cout<<"Just add new keyframe to the graph, the size of keyframe_ (without the new frame) is "<<keyframes_.size()<<"\n";
    //TODO align two functions to get another between factor

    if (keyframes_.size()>1) {
      std::list<std::shared_ptr<Frame>>::reverse_iterator rit=keyframes_.rbegin();
      rit++;
      auto kf_second_last = *rit;
      auto kf_second_last_id = kf_second_last->id;
      printf("doing map2map align between frame %d and %d\n", kf_second_last_id, last_kf->id );
      std::unique_ptr<CvoPointCloud> map_points_kf_second_last = kf_second_last->export_points_from_map();
      std::unique_ptr<CvoPointCloud> map_points_kf_last = last_kf->export_points_from_map();

      map_points_kf_second_last->write_to_label_pcd("map2map_source.pcd");
      map_points_kf_last->write_to_label_pcd("ma2map_target.pcd");

      int diff_num = std::abs(map_points_kf_last->num_points() - map_points_kf_second_last->num_points());
      
      if (diff_num * 1.0 / std::max(map_points_kf_last->num_points(), map_points_kf_second_last->num_points() ) > 0.5 || 
          ( is_f2f_ && kf_second_last_id == last_kf_id -1  )) {
        std::cout<<"the number of points in kf "<<kf_second_last_id<<" and kf "<<last_kf->id<<" differ too much: "<< map_points_kf_second_last->num_points()<<" vs "<<map_points_kf_last->num_points()<< ", or perhaps they are adjacent.  Ignore this constrains\n";
      } else {
      
        std::cout<<"Map points from the two kf exported\n"<<std::flush;
        Eigen::Affine3f init_guess = (kf_second_last->pose_in_graph().inverse() * last_kf->pose_in_graph()).inverse();
        cvo_align_.set_pcd(*map_points_kf_second_last, *map_points_kf_last,
                           init_guess, true);
        cvo_align_.align();
        Eigen::Affine3f cvo_result = cvo_align_.get_transform();
        std::cout<<"map2map transform is \n"<<cvo_result.matrix()<<std::endl;
        // TODO: check cvo align quality
        gtsam::Pose3 tf_slast_kf_to_last_kf = affine3f_to_pose3(cvo_result);
        //factor_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(kf_second_last_id ), X(last_kf_id ),
        //                                                     tf_slast_kf_to_last_kf, pose_noise));
        factor_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>((kf_second_last_id ), (last_kf_id ),
                                                             tf_slast_kf_to_last_kf, pose_noise));
        graph_values_.print("\ngraph init values\n");
        std::cout<<"Just add the edge between two  maps\n"<<std::flush;
      } 

    }
    try {
      gtsam::ISAM2Result result = isam2_->update(factor_graph_, graph_values_ ); // difference from optimize()?
      graph_values_ = isam2_->calculateEstimate();

      std::cout<<"Optimization finish\n";
      graph_values_.print("factor graph after optimization\n");
      update_optimized_poses_to_frames();
      factor_graph_.resize(0);
      graph_values_.clear();
    } catch(gtsam::IndeterminantLinearSystemException &e) {
      std::cerr<<("FORSTER2 gtsam indeterminate linear system exception!\n");
      std::cerr << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    
  }

  void PoseGraph::update_optimized_poses_to_frames() {
    //std::cout<<"graph key size: "<<graph_values_.size()<<std::endl;
    for (auto key : graph_values_.keys()) {
      //std::cout<<"key: "<<key<<". "<<std::flush;
      gtsam::Pose3 pose_gtsam= graph_values_.at<gtsam::Pose3>( key ) ;
      //std::cout<<"pose_gtsam "<<pose_gtsam<<std::endl<<std::flush;
      Mat44 pose_mat = pose_gtsam.matrix();
      Eigen::Affine3f pose;
      pose.linear() = pose_mat.block(0,0,3,3).cast<float>();
      pose.translation() = pose_mat.block(0,3,3,1).cast<float>();
      id2keyframe_[key]->set_pose_in_graph(pose);
      //std::cout<<"frame "<<key<< " new pose_in_graph is \n"<<id2keyframe_[key]->pose_in_graph().matrix()<<std::endl;
    }
    
  }

  void PoseGraph::write_trajectory(std::string filename) {
    std::ofstream outfile;;
    outfile.open(filename, std::ofstream::out );
    if (outfile.is_open()) {
      for (int i = 0; i < tracking_relative_transforms_.size(); i++) {
        Eigen::Matrix4f pose;
        if (id2keyframe_.find(i) != id2keyframe_.end()) {
          // keyframe
          auto kf = id2keyframe_[i];
          pose = kf->pose_in_graph().matrix();
        } else {
          auto ref_id = tracking_relative_transforms_[i].ref_frame_id();
          auto kf_pose = id2keyframe_[ref_id]->pose_in_graph();
          auto pose_aff = kf_pose * tracking_relative_transforms_[i].ref_frame_to_curr_frame();
          pose = pose_aff.matrix();
        }

        outfile << pose(0,0) << " "<<pose(0,1)<<" "<<pose(0,2)<<" "<<pose(0,3)<<" "
                << pose(1,0) << " "<<pose(1,1)<<" "<<pose(1,2)<<" "<<pose(1,3)<<" "
                << pose(2,0) << " "<<pose(2,1)<<" "<<pose(2,2)<<" "<<pose(2,3)
                <<"\n"<<std::flush;
        
      }
      outfile.close();
    }
  }
  
}
