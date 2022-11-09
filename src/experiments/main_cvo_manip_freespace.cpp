#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <map>
#include <boost/filesystem.hpp>
//#include "dataset_handler/KittiHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/VoxelMap.hpp"


using namespace std;
using namespace boost::filesystem;

extern template
class cvo::VoxelMap<pcl::PointXYZRGB>;

extern template
class cvo::Voxel<pcl::PointXYZRGB>;


Eigen::Vector3f get_pc_mean(const cvo::CvoPointCloud &pc) {
    Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
    for (int k = 0; k < pc.num_points(); k++)
        p_mean_tmp = (p_mean_tmp + pc.positions()[k]).eval();
    p_mean_tmp = (p_mean_tmp) / pc.num_points();
    return p_mean_tmp;
}

//constexpr auto FEATURE_DIMENSIONS = 1;
//constexpr auto NUM_CLASSES = 2;

cvo::CvoPointCloud read_pc(std::ifstream &input) {
    int num_points;
    input >> num_points;

    std::cout << "reading point cloud with " << num_points << " points\n";

    // Construct CvoPointCloud by inserting points
    cvo::CvoPointCloud pc(FEATURE_DIMENSIONS, NUM_CLASSES);
    pc.reserve(num_points, FEATURE_DIMENSIONS, NUM_CLASSES);

    for (int i = 0; i < num_points; i++) {
        Eigen::Vector3f xyz;
        float x,y,z;
        input >> x >> y >> z;
        xyz << x,y,z;

        Eigen::Matrix<float, FEATURE_DIMENSIONS, 1> feature;
        feature << 0;

        Eigen::Matrix<float, NUM_CLASSES, 1> semantics;
        int occupied;
        input >> occupied;
        if (occupied == 1) {
            semantics << 0, 1;
        } else {
            semantics << 1, 0;
        }


        // unused geometric type
        Eigen::Vector2f geometric_type;
        geometric_type << 0, 0;

//        std::cout << i << " xyz:\n" << xyz << std::endl;
//        std::cout << "feature:\n" << feature << std::endl;
//        std::cout << "semantics:\n" << semantics << std::endl;
//        std::cout << "geometric_type:\n" << geometric_type << std::endl;
        /// xyz: the 3D coordinates of the points
        /// feature: the invariant features, such as color, image gradients, etc. Its dimension is
        ///            FEATURE_DIMENSIONS. If you don't use it,
        ///            they can be assigend as zero, i.e. Eigen::VectorXf::Zero(FEATURE_DIMENSION)
        /// semantics: the semantic distribution vector, whose sum is 1. Its dimension is
        ///            NUM_CLASSES. If you don't use it, they can be assigned as zero
        /// geometric_type: A 2-dim vector, deciding whether the point is an edge or a surface.
        ///            They can be assigned as zero if you don't need this information
        pc.add_point(i, xyz, feature, semantics, geometric_type);
    }
    return pc;
}

int main(int argc, char *argv[]) {
    // list all files in current directory.
    //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
    //cvo::KittiHandler kitti(argv[1], 0);
    std::string source_file(argv[1]);
    std::string target_file(argv[2]);
    string cvo_param_file(argv[3]);
    float ell = -1;
    if (argc > 4)
        ell = std::stof(argv[4]);

    std::map<int, cvo::CvoPointCloud> poke_to_source;

    std::ifstream input_target{target_file};
    auto target = read_pc(input_target);
    // the source file will have a sequence of pokes, each followed by a point cloud
    std::ifstream input{source_file};
    while (input.peek() != EOF) {
        int poke_index = -1;
        input >> poke_index;

        std::cout << "source poke index " << poke_index << std::endl;
        // read some reasonable poke index
        if (poke_index >= 0 && poke_index < 1000) {
            poke_to_source[poke_index] = read_pc(input);
        }
    }

    // read initial transforms
    size_t lastindex = source_file.find_last_of(".");
    auto trans_file = source_file.substr(0, lastindex) + "_trans.txt";
    auto trans_gt_file = source_file.substr(0, lastindex) + "_gt_trans.txt";
    // switch to trans_gt_file for ground truth initialization
    std::ifstream input_trans{trans_file};
    int B;
    input_trans >> B;
    std::vector<Eigen::Matrix4f> guesses(B);
    for (int _b = 0; _b < B; ++_b) {
        int b;
        input_trans >> b;
        float v;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                input_trans >> v;
                guesses[b](i,j) = v;
            }
        }
    }

//    for (int b = 0; b < B; ++b) {
//        std::cout << b << std::endl;
//        std::cout << guesses[b] << std::endl;
//    }

    return 0;

    // remove file extension
    auto out_file = source_file.substr(0, lastindex) + "_cvo.txt";
    std::ofstream output{out_file};

    for (auto &pair: poke_to_source) {
        auto poke_index = pair.first;
        auto &source = pair.second;

        // TODO align
        Eigen::Vector3f source_mean = get_pc_mean(source);
        Eigen::Vector3f target_mean = get_pc_mean(target);

        float dist = (source_mean - target_mean).norm();
        std::cout << "source mean is " << source_mean << ", target mean is " << target_mean << ", dist is " << dist
                  << std::endl;
        cvo::CvoGPU cvo_align(cvo_param_file);
        cvo::CvoParams &init_param = cvo_align.get_params();
        init_param.ell_init = dist; //init_param.ell_init_first_frame;

        if (argc > 4)
            init_param.ell_init = ell;
        init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
        init_param.ell_decay_start = init_param.ell_decay_start_first_frame;
        cvo_align.write_params(&init_param);

        std::cout << "write ell! ell init is " << cvo_align.get_params().ell_init << std::endl;

        // TODO align for each batch index
        // TODO align for each poke
        Eigen::Matrix4f init_guess = guesses[0];  // from source frame to the target frame

        return 0;

        Eigen::Matrix4f result, init_guess_inv;
        Eigen::Matrix4f identity_init = Eigen::Matrix4f::Identity();
        init_guess_inv = init_guess.inverse();

        printf("Start align... num_fixed is %d, num_moving is %d\n", source.num_points(), target.num_points());
        std::cout << std::flush;

        double this_time = 0;
        cvo_align.align(source, target, init_guess_inv, result, nullptr, &this_time);

        //cvo_align.align(*source, *target, init_guess, result);

//        std::cout << "Transform is " << result << "\n\n";
//        pcl::PointCloud <pcl::PointXYZRGB> pcd_old, pcd_new;
//        cvo::CvoPointCloud new_pc(3, 19), old_pc(3, 19);
//        cvo::CvoPointCloud::transform(init_guess, target, old_pc);
//        cvo::CvoPointCloud::transform(result, target, new_pc);
//        std::cout << "Just finished transform\n";
//        cvo::CvoPointCloud sum_old = old_pc + source;
//        cvo::CvoPointCloud sum_new = new_pc + source;
//        std::cout << "Just finished CvoPointCloud concatenation\n";
//        std::cout << "num of points before and after alignment is " << sum_old.num_points() << ", "
//                  << sum_new.num_points()
//                  << "\n";
//        sum_old.export_to_pcd(pcd_old);
//        sum_new.export_to_pcd(pcd_new);
//        std::cout << "Just export to pcd\n";
//        std::string fname("before_align.pcd");
//        pcl::io::savePCDFileASCII(fname, pcd_old);
//        fname = "after_align.pcd";
//        pcl::io::savePCDFileASCII(fname, pcd_new);
        // append accum_tf_list for future initialization
        std::cout << "Average registration time is " << this_time << std::endl;
    }

//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_pcd(new pcl::PointCloud <pcl::PointXYZRGB>);
//    pcl::io::loadPCDFile(source_file, *source_pcd);
//    std::shared_ptr <cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));

//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcd(new pcl::PointCloud <pcl::PointXYZRGB>);
//    pcl::io::loadPCDFile(target_file, *target_pcd);
//    std::shared_ptr <cvo::CvoPointCloud> target(new cvo::CvoPointCloud(*target_pcd));


    return 0;
}
