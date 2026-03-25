import numpy as np
cimport pymcc_lidar
cimport numpy as np


def classify(np.ndarray[double, ndim=2, mode='c'] xyz not None,
             scaleDomain2Spacing not None,
             curvatureThreshold not None):

    m_xyz = xyz.shape[0]
    n_xyz = xyz.shape[1]

    assert n_xyz == 3

    cdef np.ndarray[double, ndim=1, mode='c'] cx = np.ascontiguousarray(xyz[:, 0])
    cdef np.ndarray[double, ndim=1, mode='c'] cy = np.ascontiguousarray(xyz[:, 1])
    cdef np.ndarray[double, ndim=1, mode='c'] cz = np.ascontiguousarray(xyz[:, 2])

    cdef int * classification
    cdef double resolution = scaleDomain2Spacing
    cdef double thresh = curvatureThreshold
    cdef int32_t n = m_xyz
    cdef Py_ssize_t i

    with nogil:
        classification = pymcc_classify(&cx[0], &cy[0], &cz[0], n, resolution, thresh)

    cdef np.ndarray[int32_t, ndim=1] np_classification = np.empty(n, dtype=np.int32)
    for i in range(n):
        np_classification[i] = classification[i]

    pymcc_free_int(classification)
    return np_classification


def calculate_excess_height(np.ndarray[double, ndim=2, mode='c'] xyz not None,
                            scaleDomainSpacing not None):

    m_xyz = xyz.shape[0]
    n_xyz = xyz.shape[1]

    assert n_xyz == 3

    cdef np.ndarray[double, ndim=1, mode='c'] cx = np.ascontiguousarray(xyz[:, 0])
    cdef np.ndarray[double, ndim=1, mode='c'] cy = np.ascontiguousarray(xyz[:, 1])
    cdef np.ndarray[double, ndim=1, mode='c'] cz = np.ascontiguousarray(xyz[:, 2])

    cdef double *h
    cdef double resolution = scaleDomainSpacing
    cdef int32_t n = m_xyz
    cdef Py_ssize_t i

    with nogil:
        h = pymcc_pass(&cx[0], &cy[0], &cz[0], n, resolution)

    cdef np.ndarray[double, ndim=1] np_h = np.empty(m_xyz)
    for i in range(m_xyz):
        np_h[i] = h[i]

    pymcc_free_double(h)
    return np_h
