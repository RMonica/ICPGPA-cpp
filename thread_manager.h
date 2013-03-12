// RMonica
#ifndef SIMPLE_THREAD_MANAGER_H
#define SIMPLE_THREAD_MANAGER_H

#include <boost/thread.hpp>

namespace NSimpleThreadManager
{

  enum ThreadAction
    {
    THREAD_ACTION_TERMINATE           = -2,
    THREAD_ACTION_NONE                = -1,
    MIN_CUSTOM_THREAD_ACTION          =  0
    };

  template <class DataType>
  class ThreadManager
    {
    public:
    typedef void (* ThreadFunc)(int thread_id,int action,DataType & data);

    explicit ThreadManager(int thread_count,ThreadFunc thread_func)
      {
      m_thread_count = thread_count - 1;
      m_action = THREAD_ACTION_NONE;
      m_running_flag.resize(m_thread_count,false);
      m_thread_func = thread_func;

      if (m_thread_count)
        {
        m_stop_conds = new boost::condition_variable[m_thread_count];
        m_threads = new boost::thread[m_thread_count];
        }

        {
        boost::mutex::scoped_lock lock(m_mutex);

        for (int i = 0; i < m_thread_count; i++)
          m_threads[i] = boost::thread(ThreadProc,this,i);
        }
      }

    ~ThreadManager()
      {
        {
        boost::mutex::scoped_lock lock(m_mutex);

        m_action = THREAD_ACTION_TERMINATE;
        for (int i = 0; i < m_thread_count; i++)
          m_running_flag[i] = true;

        m_start_cond.notify_all();
        }

      for (int i = 0; i < m_thread_count; i++)
        m_threads[i].join();

      if (m_thread_count)
        {
        delete [] m_threads;
        delete [] m_stop_conds;
        }
      }

    static void ThreadProc(ThreadManager * info,int thread_id)
      {
      while (true)
        {
        // scope only
          {
          boost::mutex::scoped_lock lock(info->m_mutex);

          if (!info->m_running_flag[thread_id])
            info->m_start_cond.wait(lock);
          }

        switch (info->m_action)
          {
          case THREAD_ACTION_NONE:
            break;
          case THREAD_ACTION_TERMINATE:
            return;
          default:
            info->m_thread_func(thread_id,info->m_action,info->m_data);
            break;
          }

        // scope only
          {
          boost::mutex::scoped_lock lock(info->m_mutex);
          info->m_running_flag[thread_id] = false;
          info->m_stop_conds[thread_id].notify_one();
          }
        }
      }

    void ExecuteAction(int action)
      {
      if (action < MIN_CUSTOM_THREAD_ACTION)
        return; // action not allowed

      // scope only
        {
        boost::mutex::scoped_lock lock(m_mutex);

        m_action = action;
        for (int i = 0; i < m_thread_count; i++)
          m_running_flag[i] = true;

        m_start_cond.notify_all();
        }

      // use the main thread to simulate thread m_thread_count
      m_thread_func(m_thread_count,m_action,m_data);

      // scope only
        {
        boost::mutex::scoped_lock lock(m_mutex);

        for (int i = 0; i < m_thread_count; i++)
          {
          if (m_running_flag[i])
            m_stop_conds[i].wait(lock);
          }
        }
      }

    DataType & Data() {return m_data; }

    int GetThreadCount() {return m_thread_count + 1; }

    private:
    // forbid empty and copy constructors
    ThreadManager() {}
    ThreadManager(const ThreadManager &) {}

    int m_thread_count;

    int m_action;
    std::vector<bool> m_running_flag;

    boost::mutex m_mutex;
    boost::condition_variable m_start_cond;
    boost::condition_variable * m_stop_conds;

    boost::thread * m_threads;

    ThreadFunc m_thread_func;

    DataType m_data;
    };

  /// Subdivides a task into spans
  /// that can then be distributed among threads
  /// @param task_length the total number of operations of the task
  /// @param span_id the span that must be calculated (0 based)
  /// @param span_count the total number of spans
  /// @returns the id of the first operation in this span (0 based)
  ///          or task_length if span_id >= span_count
  /// @note: no values <= 0 allowed!
  inline int GetTaskSpanFirst(int task_length,int span_id,int span_count)
    {
    if (span_id >= span_count)
      return task_length;

    return task_length * span_id / span_count;
    }

} // NSimplethreadManager

#endif
