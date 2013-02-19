// RMonica
#ifndef ICPGPA_REGISTRATION_H_
#define ICPGPA_REGISTRATION_H_

// EIGEN
#include <Eigen/SVD>

// STL
#include <vector>
#include <deque>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/eigen.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>

#include "thread_manager.h"

namespace pcl
{

template <typename PointT>
class ICPGPARegistration
  {
  public:
  typedef float Real;

  typedef pcl::PointCloud<PointT> PointCloud;
  typedef boost::shared_ptr<PointCloud> PointCloudPtr;
  typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
  typedef std::vector<PointCloudPtr> PointCloudPtrVector;
  typedef std::vector<PointCloudConstPtr> PointCloudConstPtrVector;

  typedef Eigen::Matrix<Real,3,3> RotationMatrix;
  typedef Eigen::Matrix<Real,3,1> TranslationVector;

  typedef Eigen::Affine3f TransformationMatrix;
  typedef std::vector<TransformationMatrix> TransformationMatrixVector;

  typedef Real ScaleFactor;

  typedef Eigen::Matrix<Real,3,Eigen::Dynamic> DynamicPointMatrix;
  typedef Eigen::Matrix<Real,3,1> PointVector;

  typedef Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic> DynamicMatrix;
  typedef Eigen::Matrix<Real,3,3> Matrix3x3;

  typedef Eigen::Matrix<Real,1,Eigen::Dynamic> DynamicRowVector;
  typedef Eigen::Matrix<Real,Eigen::Dynamic,1> DynamicColVector;

  // this vector contains the neighboring point indices
  // for a single point
  // the index is the cloud of the indexed point
  typedef std::vector<int> PointNeighVector;
  // this vector contains the neighbour relations
  // for all the points in a single cloud
  // the index is the index of the point in the cloud
  typedef std::vector<PointNeighVector> PointNeighCloud;
  // this vector contains the neighbour relations
  // for all the points
  // the index is the cloud of the source point
  typedef std::vector<PointNeighCloud> PointNeighGraph;

  typedef KdTreeFLANN<PointT> KdTree;
  typedef boost::shared_ptr<KdTree> KdTreePtr;

  typedef std::pair<int,int> IntPair;

  typedef ICPGPARegistration<PointT> SelfType;

  typedef NSimpleThreadManager::ThreadManager<SelfType *> ThreadManager;

  /// This class can be subclassed to override the default weights
  /// assigned to each set of mutual neighboring points
  class WeightFunc
    {
    public:
    /// Computes the weight associated to the set
    /// i. e. calculates the likelihood that the association
    /// described in the set is correct
    /// WARNING: may be called by multiple thread simultaneously!
    /// @param clouds the clouds
    /// @param cloud_idx the specific cloud for which the weight must be calculated
    /// @param set the set
    /// @returns a real value between 0.0 and 1.0
    virtual Real ComputeWeight(const PointCloudPtrVector & clouds,int cloud_idx,
      const PointNeighVector & set) = 0;
    };

  typedef boost::shared_ptr<WeightFunc> WeightFuncPtr;

  /// This class can be subclassed to receive events
  /// while the object is processing
  /// All the event handlers will be called by the main thread
  /// (i. e. the thread that called the method "process")
  class Listener
    {
    public:
    virtual void onStartIteration(int /*iteration*/) {}
    virtual void onEndIteration(int /*iteration*/) {}

    virtual void onComputedGraph(int /*iteration*/,const PointNeighGraph & /*graph*/) {}
    virtual void onFoundIndependentSets(int /*iteration*/,const PointNeighCloud & /*sets*/) {}
    virtual void onComputedCentroids(int /*iteration*/,const DynamicPointMatrix & /*centroids*/) {}

    virtual void onTransformedCloud(int /*iteration*/,int /*cloudid*/,const TransformationMatrix & /*matrix*/,
      const PointCloud & /*newcloud*/) {}
    };

  typedef boost::shared_ptr<Listener> ListenerPtr;

  /// this class is instantiated internally and contains the pointers to the listeners
  /// can be accessed and modified by reference using getListenerContainer()
  class ListenerContainer: public Listener, public std::vector<ListenerPtr>
    {
    public:
    using typename std::vector<ListenerPtr>::size_type;

    virtual void onStartIteration(int iteration)
      {for (size_type i = 0; i < this->size(); i++) (*this)[i]->onStartIteration(iteration); }
    virtual void onEndIteration(int iteration)
      {for (size_type i = 0; i < this->size(); i++) (*this)[i]->onEndIteration(iteration); }

    virtual void onComputedGraph(int iteration,const PointNeighGraph & graph)
      {for (size_type i = 0; i < this->size(); i++) (*this)[i]->onComputedGraph(iteration,graph); }
    virtual void onFoundIndependentSets(int iteration,const PointNeighCloud & sets)
      {for (size_type i = 0; i < this->size(); i++) (*this)[i]->onFoundIndependentSets(iteration,sets); }
    virtual void onComputedCentroids(int iteration,const DynamicPointMatrix & centroids)
      {for (size_type i = 0; i < this->size(); i++) (*this)[i]->onComputedCentroids(iteration,centroids); }

    virtual void onTransformedCloud(int iteration,int cloudid,const TransformationMatrix & matrix,
      const PointCloud & newcloud)
      {for (size_type i = 0; i < this->size(); i++) (*this)[i]->onTransformedCloud(iteration,cloudid,matrix,newcloud); }
    };

  enum ThreadActions
    {
    THREAD_ACTION_FIRST  = NSimpleThreadManager::MIN_CUSTOM_THREAD_ACTION,
    THREAD_ACTION_COMPUTE_POINT_GRAPH = THREAD_ACTION_FIRST + 0,
    THREAD_ACTION_INIT_KDTREES        = THREAD_ACTION_FIRST + 1,
    THREAD_ACTION_PROCESS_CLOUD_SPAN  = THREAD_ACTION_FIRST + 2
    };

  /// main constructor
  /// @param num_of_clouds the number of clouds that
  ///        will be processed by this class
  /// @param thread_count the number of threads that can be created
  ///        (default 1, >= 1)
  ICPGPARegistration(int num_of_clouds,int thread_count = 1):
    m_thread_manager(thread_count,ThreadFunction)
    {
    m_num_of_clouds = num_of_clouds;

    m_euclidean_threshold = -1.0;

    m_thread_manager.Data() = this;

    // initialize with null transformation
    m_transformations.resize(num_of_clouds,TransformationMatrix::Identity());
    m_last_transformations.resize(num_of_clouds,TransformationMatrix::Identity());

    m_clouds.resize(num_of_clouds);
    }

  ~ICPGPARegistration()
    {

    }

  /// get the number of clouds
  /// @returns the cloud count
  int getCloudCount() const
    {
    return m_num_of_clouds;
    }

  /// sets all the clouds at once
  /// @param clouds the new clouds
  ///        must be of the same size as num_of_clouds
  void setClouds(const PointCloudPtrVector clouds)
    {
    if (int(clouds.size()) != m_num_of_clouds)
      return; // invalid

    m_clouds = clouds;
    }

  /// set the cloud at an index
  /// WARNING: the cloud will be changed while processing
  ///          so it acts like both an output and an input cloud
  /// @param i the index (0 based)
  /// @param cloud the cloud
  void setCloud(int i,const PointCloudPtr cloud)
    {
    if (i >= m_num_of_clouds || i < 0)
      return;

    m_clouds[i] = cloud;
    }

  /// gets the cloud at a certain index
  /// @param i the index
  /// @returns the cloud at index i
  const PointCloudPtr getCloud(int i) const
    {
    return m_clouds[i];
    }

  /// gets all the clouds at once
  /// @returns a 0 based vector with all the clouds
  const PointCloudPtrVector & getClouds() const
    {
    return m_clouds;
    }

  /// gets the transformation matrix of a cloud after processing
  /// @param i the index
  /// @returns an affine matrix
  const TransformationMatrix & getTransformation(int i) const
    {
    return m_transformations[i];
    }

  /// gets all the transformation matrices after processing
  /// @returns a vector of affine matrices
  const TransformationMatrixVector & getTransformations() const
    {
    return m_transformations;
    }

  /// get the transformations applied during last step
  /// @returns a vector of affine matrices
  const TransformationMatrixVector & getLastTransformations() const
    {
    return m_last_transformations;
    }

  /// get the transformation applied during last step to a single cloud
  /// @param cloud the cloud id
  /// @returns an affine matrix
  const TransformationMatrix & getLastTransformation(int cloud) const
    {
    return m_last_transformations[cloud];
    }

  /// Get the graph calculated during last iteration
  /// @returns the graph
  const PointNeighGraph & getLastGraph() const
    {
    return m_graph;
    }

  /// Get the sets calculated during last iteration
  /// @returns the sets
  const PointNeighCloud & getLastSets() const
    {
    return m_sets;
    }

  /// Get the centroids calculated during last iteration
  /// @returns a matrix with 3 rows (x,y,z) and a column for each centroid
  const DynamicPointMatrix & getLastCentroids() const
    {
    return m_centroids;
    }

  /// Computes the mean square error after the last iteration
  /// @returns the error
  Real getMeanSquareError() const
    {
    return ComputeMeanSquareError(m_clouds,m_sets,m_centroids);
    }

  /// Set a maximum distance for point matches
  /// point matches with higher distance will be discarded
  /// @param dist the maximum distance
  ///             if negative, the check will be ignored
  /// default value is -1.0 (check ignored)
  void setMaxMatchDistance(Real dist)
    {
    m_euclidean_threshold = dist;
    }

  /// Returns a reference to the internal listener container
  ListenerContainer & getListenerContainer()
    {
    return m_listeners;
    }

  /// Pass a subclass of WeightFunc to this function
  /// to override the default weight of 1.0 for each
  /// point correnspondance
  /// @param func a WeightFuncPtr to the class
  ///             or WeightFuncPtr(NULL) to unset
  void setWeightFunction(WeightFuncPtr func)
    {
    m_weight_func = func;
    }

  /// starts or advance processing
  /// @param steps the number of cycles
  void process(int steps)
    {
    if (steps <= 0)
      return;

    for (int step = 0; step < steps; step++)
      {
      m_listeners.onStartIteration(step);

      ComputePointGraph(m_clouds,m_graph,m_trees,m_euclidean_threshold,m_thread_manager);
      m_listeners.onComputedGraph(step,m_graph);

      FindIndependentSets(m_graph,m_sets);
      m_listeners.onFoundIndependentSets(step,m_sets);

      ComputeCentroids(m_clouds,m_sets,m_centroids);
      m_listeners.onComputedCentroids(step,m_centroids);

      m_thread_manager.ExecuteAction(THREAD_ACTION_PROCESS_CLOUD_SPAN);

      for (int i = 0; i < m_num_of_clouds; i++)
        m_listeners.onTransformedCloud(step,i,m_last_transformations[i],*(m_clouds[i]));
      m_listeners.onEndIteration(step);
      }
    }

  static const int NO_POINT = -1;

  /// Executes the logic of ComputePointGraph for only a part of the points
  /// @param clouds the clouds
  /// @param graph the partly produced graph
  /// @param trees the search trees that may be used
  /// @param span_id the part of the job that must be calculated
  /// @param total_spans the total number of parts into which the job is subdivided
  static void ComputePointGraphSpan(const PointCloudPtrVector & clouds,PointNeighGraph & graph,
    const std::vector<KdTreePtr> & trees,Real euclidean_threshold,int span_id,int total_spans)
    {
    bool need_threshold = euclidean_threshold >= 0.0;
    Real sqr_threshold = euclidean_threshold * euclidean_threshold;

    int cloud_count = clouds.size();
    for (int source_cloud_idx = 0; source_cloud_idx < cloud_count; source_cloud_idx++)
      {
      int source_cloud_size = clouds[source_cloud_idx]->size();
      int from_point_idx = NSimpleThreadManager::GetTaskSpanFirst(source_cloud_size,span_id,total_spans);
      int to_point_idx = NSimpleThreadManager::GetTaskSpanFirst(source_cloud_size,span_id + 1,total_spans);
      for (int dest_cloud_idx = source_cloud_idx + 1; dest_cloud_idx < cloud_count; dest_cloud_idx++)
        {
        for (int src_point_idx = from_point_idx; src_point_idx < to_point_idx; src_point_idx++)
          {
          int dest_point_idx;
          int r;

          std::vector<int> resultvec(1,0);
          std::vector<float> sqr_dist(1,0.0);
          // do the KdSearch
          r = trees[dest_cloud_idx]->nearestKSearch(*(clouds[source_cloud_idx]),src_point_idx,
            1,resultvec,sqr_dist);
          if (!r)
            continue; // something went wrong
          if (need_threshold && sqr_dist[0] > sqr_threshold)
            continue; // points too far
          dest_point_idx = resultvec[0];

          // check if the points are mutual nearest neighbors
          r = trees[source_cloud_idx]->nearestKSearch(*(clouds[dest_cloud_idx]),dest_point_idx,
            1,resultvec,sqr_dist);
          if (!r)
            continue; // something went wrong
          if (resultvec[0] != src_point_idx)
            continue; // not mutual

          graph[source_cloud_idx][src_point_idx][dest_cloud_idx] = dest_point_idx;
          graph[dest_cloud_idx][dest_point_idx][source_cloud_idx] = src_point_idx;
          }
        }
      }
    }

  /// Computes and applies the transformation to a subset of the clouds
  /// @param clouds the clouds
  /// @param sets the sets
  /// @param centroids the centroids
  /// @param weights the weights (or NULL if none)
  /// @param cur_transform the produced and applied current transformations
  /// @param total_transforms the transformations applied from the beginning
  /// @param span_id the part of the job that must be done
  /// @param total_spans the total number of parts into which the job is subdivided
  static void TransformCloudSpan(PointCloudPtrVector & clouds,const PointNeighCloud & sets,const DynamicPointMatrix centroids,
    const WeightFuncPtr & weight_func,TransformationMatrixVector & cur_transforms,
    TransformationMatrixVector & total_transforms,int span_id,int total_spans)
    {
    int cloud_count = clouds.size();
    int from_cloud_idx = NSimpleThreadManager::GetTaskSpanFirst(cloud_count,span_id,total_spans);
    int to_cloud_idx = NSimpleThreadManager::GetTaskSpanFirst(cloud_count,span_id + 1,total_spans);

    DynamicPointMatrix local_centroids;
    DynamicPointMatrix local_points;
    DynamicColVector local_weights;

    for (int cloud_idx = from_cloud_idx; cloud_idx < to_cloud_idx; cloud_idx++)
      {
      ComputeCentroidsForCloud(clouds,sets,cloud_idx,centroids,local_points,local_centroids);
      if (weight_func)
        ComputeWeights(clouds,sets,cloud_idx,weight_func,local_weights);

      cur_transforms[cloud_idx] = weight_func ? PointsToCentroidsWeighted(local_points,local_centroids,local_weights) :
        PointsToCentroids(local_points,local_centroids);
      pcl::transformPointCloud(*(clouds[cloud_idx]),*(clouds[cloud_idx]),cur_transforms[cloud_idx]);
      total_transforms[cloud_idx] = cur_transforms[cloud_idx] * total_transforms[cloud_idx];
      }
    }

  /// Initializes a subset of the KdTrees
  /// @param clouds the clouds
  /// @param trees the trees
  /// @param span_id the part of the job that must be done
  /// @param total_spans the total number of parts into which the job is subdivided
  static void InitKdTreeSpan(const PointCloudPtrVector & clouds,std::vector<KdTreePtr> & trees,
    int span_id,int total_spans)
    {
    int tree_count = trees.size();
    int from_tree_idx = NSimpleThreadManager::GetTaskSpanFirst(tree_count,span_id,total_spans);
    int to_tree_idx = NSimpleThreadManager::GetTaskSpanFirst(tree_count,span_id + 1,total_spans);

    for (int i = from_tree_idx; i < to_tree_idx; i++)
      {
      trees[i] = KdTreePtr(new KdTree());
      trees[i]->setInputCloud(clouds[i]);
      }
    }

  /// compute the mutual nearest neighborhood relation graph
  ///
  /// For each cloud and for each point in that cloud
  /// a vector of int is produced, with one element for each cloud.
  /// Each element contains the index of the mutual nearest neighbor
  /// in the cloud for the point, or NO_POINT if none found.
  /// NOTE: a point is not the nearest neighbor of itself,
  ///       so graph[a][b][a] is always NO_POINT.
  /// @param clouds the clouds
  /// @param graph the graph produced
  static void ComputePointGraph(const PointCloudPtrVector & clouds,PointNeighGraph & graph,
    std::vector<KdTreePtr> &trees,Real euclidean_threshold,ThreadManager & thread_manager)
    {
    const int cloud_count = clouds.size();
    if (cloud_count <= 0)
      return;

    // reinitialize the graph
    graph.resize(cloud_count);
    for (int i = 0; i < cloud_count; i++)
      {
      int cloudsize = clouds[i]->size();
      graph[i].resize(cloudsize);
      for (int h = 0; h < cloudsize; h++)
        {
        graph[i][h].resize(cloud_count);
        for (int k = 0; k < cloud_count; k++)
          graph[i][h][k] = NO_POINT;
        }
      }

    // create the KdTrees for nearest neighbor search
    trees.resize(cloud_count);
    thread_manager.ExecuteAction(THREAD_ACTION_INIT_KDTREES);

    thread_manager.ExecuteAction(THREAD_ACTION_COMPUTE_POINT_GRAPH);
    }

  /// Finds the independent sets of mutual neighbor point
  /// excluding: sets with only one point
  ///            and set with more than one point of the same cloud
  /// Produces an array of sets
  /// each set is represented by a vector of integer with num_of_clouds elements
  /// the element at position i is the index of the point
  /// that belongs to that set for cloud i, or NO_POINT if none found
  /// @param graph the neighborhood graph
  /// @param sets the array of produced sets (cleared at the beginning)
  static void FindIndependentSets(const PointNeighGraph & graph,PointNeighCloud & sets)
    {
    // using floodfill labelling for sets

    const int cloud_count = graph.size();

    // stores if the point at [cloud][index] has already been scanned
    std::vector<std::vector<bool> > scanned;
    // initialize to false
    scanned.resize(cloud_count);
    for (int i = 0; i < cloud_count; i++)
      scanned[i].resize(graph[i].size(),false);

    sets.clear();
    // approximate the number of sets with this, for performance
    sets.reserve(graph[0].size());

    // the set will be temporarily built here
    PointNeighVector tempSet(cloud_count,NO_POINT);

    // this is the queue for floodfill labelling
    // points are a pair <cloud_idx,point_idx>
    std::deque<IntPair> scanQueue;

    for (int cloud_idx = 0; cloud_idx < cloud_count; cloud_idx++)
      {
      const int point_count = graph[cloud_idx].size();
      for (int point_idx = 0; point_idx < point_count; point_idx++)
        {
        // find the next not scanned point
        if (!scanned[cloud_idx][point_idx])
          {
          scanned[cloud_idx][point_idx] = true;

          // points are useless unless they have neighborhood
          bool isUseful = false;

          for (int i = 0; i < cloud_count; i++)
            tempSet[i] = NO_POINT;
          // the set includes this point
          tempSet[cloud_idx] = point_idx;

          int tpx;

          // scan the neighborhood of the point
          for (int i = 0; i < cloud_count; i++)
            if ((tpx = graph[cloud_idx][point_idx][i]) != NO_POINT)
              {
              scanned[i][tpx] = true;
              // found at least a neighbor: the set is useful
              isUseful = true;
              // add the point to the set
              tempSet[i] = tpx;
              //schedule its neighborhood for scan
              scanQueue.push_back(IntPair(i,tpx));
              }

          // scan the neigborhood of the neigborhood
          while (!scanQueue.empty())
            {
            IntPair point_idx_pair = scanQueue.front();
            scanQueue.pop_front();
            for (int i = 0; i < cloud_count; i++)
              if ((tpx = graph[point_idx_pair.first][point_idx_pair.second][i]) != NO_POINT &&
                !scanned[i][tpx])
                {
                scanned[i][tpx] = true;

                if (tempSet[i] == NO_POINT)
                  tempSet[i] = tpx;
                  else
                    // if more than one point of the same cloud are
                    // in the set, the set is no more useful
                    isUseful = false;
                    // but continue scanning: the neighbor of a
                    // useless point is useless, so we can exclude
                    // more points from future scans

                scanQueue.push_back(IntPair(i,tpx));
                }
            }

          if (isUseful)
            sets.push_back(tempSet);
          }
        }
      }
    }

  /// convert a point from PointT to PointVector
  static PointVector PointTToEigen(const PointT & point)
    {
    return PointVector(point.x,point.y,point.z);
    }

  /// Computes the centroids for the sets
  /// produces a matrix composed by a column vector
  /// for each set, in the same order
  /// @param clouds the clouds
  /// @param sets the sets
  /// @param centroids the output matrix
  static void ComputeCentroids(const PointCloudPtrVector & clouds,const PointNeighCloud & sets,
    DynamicPointMatrix & centroids)
    {
    const int set_count = sets.size();
    const int cloud_count = clouds.size();

    centroids.resize(Eigen::NoChange,set_count);

    // TODO: make this multithreaded, maybe
    for (int set_idx = 0; set_idx < set_count; set_idx++)
      {
      centroids.col(set_idx) = PointVector::Zero();
      int count = 0;
      int tpx;
      for (int cloud_idx = 0; cloud_idx < cloud_count; cloud_idx++)
        if ((tpx = sets[set_idx][cloud_idx]) != NO_POINT)
          {
          centroids.col(set_idx) += PointTToEigen(clouds[cloud_idx]->points[tpx]);
          count++;
          }

      centroids.col(set_idx) /= Real(count);
      }
    }

  /// Produces two matrices, one containing the interesting points of the cloud
  /// and the other the centroids to which the points should be moved
  /// @param clouds the clouds (only cloud cloud_idx will be used)
  /// @param sets the independent sets
  /// @param cloud_idx the index of the cloud in clouds that must be used
  /// @param centroids the centroid matrix, one for each set in sets
  /// @param local_points the interesting points
  /// @param local_centroids the corresponding centroids
  static void ComputeCentroidsForCloud(const PointCloudPtrVector & clouds,const PointNeighCloud & sets,
    const int cloud_idx,const DynamicPointMatrix & centroids,
    DynamicPointMatrix & local_points,DynamicPointMatrix & local_centroids)
    {
    const int set_count = sets.size();

    // the amount of points produced is limited from above
    // by both the number of sets
    // and the number of points in the cloud
    // because a point can not be in two different sets
    int estimated_count = clouds[cloud_idx]->size();
    if (set_count < estimated_count)
      estimated_count = set_count;
    local_centroids.resize(Eigen::NoChange,estimated_count);
    local_points.resize(Eigen::NoChange,estimated_count);

    int counter = 0;

    int tpx;

    for (int set_idx = 0; set_idx < set_count; set_idx++)
      if ((tpx = sets[set_idx][cloud_idx]) != NO_POINT)
        {
        local_centroids.col(counter) = centroids.col(set_idx);
        local_points.col(counter) = PointTToEigen(clouds[cloud_idx]->points[tpx]);
        counter++;
        }

    // resize to the actual size
    local_centroids.conservativeResize(Eigen::NoChange,counter);
    local_points.conservativeResize(Eigen::NoChange,counter);
    }

  /// Produces the a weight for each set
  /// @param clouds the clouds
  /// @param sets the sets
  /// @param cloud_idx the specific cloud for which the weights must be computed
  /// @param weightfunc the pointer to the weight function
  /// @param weights the produced weights
  static void ComputeWeights(const PointCloudPtrVector & clouds,const PointNeighCloud & sets,
    const int cloud_idx,const WeightFuncPtr & weightfunc,DynamicColVector & weights)
    {
    const int set_count = sets.size();

    // estimate
    weights.resize(set_count);

    int actual_count = 0;

    for (int set_idx = 0; set_idx < set_count; set_idx++)
      if (sets[set_idx][cloud_idx] != NO_POINT)
        {
        weights[actual_count] = weightfunc->ComputeWeight(clouds,cloud_idx,sets[set_idx]);
        actual_count++;
        }

    // actual resize
    weights.conservativeResize(actual_count);
    }

  /// Computes the Mean Square distance between the centroids
  /// and the corresponding points for all the clouds
  /// @param clouds the clouds
  /// @param sets the independent sets
  /// @param centroids the centroids, one for each set
  static Real ComputeMeanSquareError(const PointCloudPtrVector & clouds,const PointNeighCloud & sets,
    const DynamicPointMatrix & centroids)
    {
    Real error = 0.0;

    int cloud_count = clouds.size();
    int set_count = sets.size();

    if (!cloud_count || !set_count)
      return error; // empty for some reason

    int tpx;

    int counter = 0;

    PointVector diff;

    for (int set_idx = 0; set_idx < set_count; set_idx++)
      for (int cloud_idx = 0; cloud_idx < cloud_count; cloud_idx++)
        if ((tpx = sets[set_idx][cloud_idx]) != NO_POINT)
          {
          diff = PointTToEigen(clouds[cloud_idx]->points[tpx]) - centroids.col(set_idx);
          // dot product to compute square distance
          error += diff.transpose() * diff;
          counter++;
          }

    if (counter != 0)
      error /= Real(counter);
    return error;
    }

  /// computes the best possible affine transformation to bring points in points
  /// to the centroid centroids
  /// the two parameters must be of the same size
  /// @param points the points (column vectors) (one point each column)
  /// @param centroids the target positions (column vectors) (one point each column)
  /// @returns the transformation matrix
  static TransformationMatrix PointsToCentroids(const DynamicPointMatrix & points,
    const DynamicPointMatrix & centroids)
    {
    // procrustes

    // no matches found, return identity
    int point_count = points.cols();
    if (!point_count)
      return Eigen::Affine3f::Identity();

    // compute the mean centroid
    PointVector mean_centroid = PointVector::Zero();
    for (int i = 0; i < point_count; i++)
      mean_centroid += centroids.col(i);
    mean_centroid /= Real(point_count);

    // compute the mean point
    PointVector mean_point = PointVector::Zero();
    for (int i = 0; i < point_count; i++)
      mean_point += points.col(i);
    mean_point /= Real(point_count);

    // translate the points by the mean point
    DynamicPointMatrix shifted_points = points;
    for (int i = 0; i < point_count; i++)
      shifted_points.col(i) -= mean_point;

    // translate the centroids by the mean centroid
    DynamicPointMatrix shifted_centroids = centroids;
    for (int i = 0; i < point_count; i++)
      shifted_centroids.col(i) -= mean_centroid;

    // compute K
    Matrix3x3 K = shifted_points * shifted_centroids.transpose();
    K /= Real(point_count);

    // procrustes: use SVD to get eigenvectors
    Eigen::JacobiSVD<Matrix3x3> svd_solver = K.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix3x3 U = svd_solver.matrixU();
    Matrix3x3 V = svd_solver.matrixV();

    // find the rotation from points to centroids
    RotationMatrix rotation = V * U.transpose();

    // correct if the rotation determinant is -1 instead of 1
    Real detR = rotation.determinant();
    if (detR < 0)
      {
      Matrix3x3 id = Matrix3x3::Identity();
      id(2,2) = detR;
      rotation = V * id * U.transpose();
      }

    // find the translation subtracting the mean point and centroid
    PointVector translation = mean_centroid - (rotation * mean_point);

    // create transformation
    TransformationMatrix result = TransformationMatrix::Identity();
    result.translate(translation);
    result.rotate(rotation);

    return result;
    }

  /// computes the best possible affine transformation to bring points in points
  /// to the centroid centroids using the weights
  /// the two parameters must be of the same size
  /// @param points the points (column vectors) (one point each column)
  /// @param centroids the target positions (column vectors) (one point each column)
  /// @param weights the weights that must be assigned to each correspondance
  /// @returns the transformation matrix
  static TransformationMatrix PointsToCentroidsWeighted(const DynamicPointMatrix & points,
    const DynamicPointMatrix & centroids,const DynamicColVector & weights)
    {
    // weighted procrustes

    // no matches found, return identity
    int point_count = points.cols();
    if (!point_count)
      return Eigen::Affine3f::Identity();

    Real weights_sqr_norm = weights.squaredNorm();
    if (weights_sqr_norm <= 0.0)
      return Eigen::Affine3f::Identity(); // no non-0 weights found

    DynamicColVector normalized_weights = weights / weights_sqr_norm;

    DynamicPointMatrix AjMul;
    AjMul.resize(points.rows(),point_count);

    for (int i = 0; i < point_count; i++)
      {
      DynamicRowVector col = - weights[i] * normalized_weights.transpose();
      col[i] += 1.0;

      AjMul.col(i) = points * col.transpose();
      }

    Matrix3x3 K = AjMul * centroids.transpose();

    Eigen::JacobiSVD<Matrix3x3> svd_solver = K.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix3x3 U = svd_solver.matrixU();
    Matrix3x3 V = svd_solver.matrixV();

    RotationMatrix rotation = V * U.transpose();
    ScaleFactor scale = (rotation.transpose() * AjMul * centroids.transpose()).trace() /
      (AjMul * points.transpose()).trace();
    TranslationVector translation = (centroids - (scale * rotation * points)) * normalized_weights;

    TransformationMatrix result = TransformationMatrix::Identity();
    result.translate(translation);
    result.scale(scale);
    result.rotate(rotation);

    return result;
    }

  int GetThreadCount() {return m_thread_manager.GetThreadCount(); }

  private:
  static void ThreadFunction(int thread_id,int action,SelfType * &data)
    {
    switch (action)
      {
      case THREAD_ACTION_COMPUTE_POINT_GRAPH:
        ComputePointGraphSpan(data->m_clouds,data->m_graph,data->m_trees,data->m_euclidean_threshold,
          thread_id,data->GetThreadCount());
        break;
      case THREAD_ACTION_INIT_KDTREES:
        InitKdTreeSpan(data->m_clouds,data->m_trees,thread_id,data->GetThreadCount());
        break;
      case THREAD_ACTION_PROCESS_CLOUD_SPAN:
        TransformCloudSpan(data->m_clouds,data->m_sets,data->m_centroids,data->m_weight_func,
        data->m_last_transformations,data->m_transformations,thread_id,data->GetThreadCount());
        break;
      default:
        break;
      }
    }

  TransformationMatrixVector m_transformations;
  TransformationMatrixVector m_last_transformations;

  PointNeighGraph m_graph;
  PointNeighCloud m_sets;
  DynamicPointMatrix m_centroids;

  PointCloudPtrVector m_clouds;

  std::vector<KdTreePtr> m_trees;

  int m_num_of_clouds;

  Real m_euclidean_threshold;

  ThreadManager m_thread_manager;

  ListenerContainer m_listeners;

  WeightFuncPtr m_weight_func;
  };

} // pcl

#endif // ICPGPA_REGISTRATION_H_
