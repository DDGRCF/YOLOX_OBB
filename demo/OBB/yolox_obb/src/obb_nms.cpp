#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

#define ax(i) i * 2
#define ay(i) i * 2 + 1
#define OBB 0
#define Poly 1

template <typename T>
struct Point {
  T x, y;
   Point(const T& px = 0, const T& py = 0) : x(px), y(py) {}
   Point operator+(const Point& p) const {
    return Point(x + p.x, y + p.y);
  }
   Point& operator+=(const Point& p) {
    x += p.x;
    y += p.y;
    return *this;
  }
   Point operator-(const Point& p) const {
    return Point(x - p.x, y - p.y);
  }
   Point operator*(const T coeff) const {
    return Point(x * coeff, y * coeff);
  }
};


template <typename T>
 T dot_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.x + A.y * B.y;
}

// R: result type. can be different from input type
template <typename T, typename R = T>
 R cross_2d(const Point<T>& A, const Point<T>& B) {
  return static_cast<R>(A.x) * static_cast<R>(B.y) -
      static_cast<R>(B.x) * static_cast<R>(A.y);
}

template <typename T>
 int get_intersection_points(
    const Point<T> (&pts1)[4],
    const Point<T> (&pts2)[4],
    Point<T> (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  int num = 0; // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      T det = cross_2d<T>(vec2[j], vec1[i]);

      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      T t1 = cross_2d<T>(vec2[j], vec12) / det;
      T t2 = cross_2d<T>(vec1[i], vec12) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto& AB = vec1[0];
    const auto& DA = vec1[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

template <typename T>
 int convex_hull_graham(
    const Point<T> (&p)[24],
    const int& num_in,
    Point<T> (&q)[24],
    bool shift_to_zero = false) {
  assert(num_in >= 2);

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto& start = p[t]; // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  T dist[24];
  // CPU version
  std::sort(
      q + 1, q + num_in, [](const Point<T>& A, const Point<T>& B) -> bool {
        T temp = cross_2d<T>(A, B);
        if (fabs(temp) < 1e-6) {
          return dot_2d<T>(A, A) < dot_2d<T>(B, B);
        } else {
          return temp > 0;
        }
      });
  // compute distance to origin after sort, since the points are now different.
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }

  int k; // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2; // 2 points in the stack
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1) {
      auto q1 = q[i] - q[m - 2], q2 = q[m - 1] - q[m - 2];
      if (q1.x * q2.y >= q2.x * q1.y)
        m--;
      else
        break;
    }
    q[m++] = q[i];
  }

  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }

  return m;
}

template <typename T>
 T polygon_area(const Point<T> (&q)[24], const int& m) {
  if (m <= 2) {
    return 0;
  }

  T area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross_2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

template <typename T>
T polygon_area(T const * const box, const int & m) {
    T area{0};
    for (int i = 0; i < m; i++) {
        area += (i != m-1) ? (box[ax(i)] * box[ay(i + 1)] - box[ay(i)] * box[ax(i + 1)]) \
            : (box[ax(i)] * box[ay(0)] - box[ay(i)] * box[ax(0)]);
    }
    return area;
}

template <typename T>
 T rotated_boxes_intersection(
    const Point<T> (&pts1)[4],
    const Point<T> (&pts2)[4],
    const int & nms_type) {
    Point<T> intersectPts[24], orderedPts[24];

    int num = get_intersection_points<T>(pts1, pts2, intersectPts);

    if (num <= 2) {
        return 0.0;
    }

    int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
    return polygon_area<T>(orderedPts, num_convex);
}


template <typename T>
 void get_rotated_vertices(
    T const * const box,
    Point<T> (&pts)[4],
    const int & nms_type) {

  if (nms_type == OBB) {
    double theta = box[4];
    T cosTheta2 = (T)cos(theta) * 0.5f;
    T sinTheta2 = (T)sin(theta) * 0.5f;
    // y: top --> down; x: left --> right
    pts[0].x = box[0] + sinTheta2 * box[3] + cosTheta2 * box[2];
    pts[0].y = box[1] + cosTheta2 * box[3] - sinTheta2 * box[2];
    pts[1].x = box[0] - sinTheta2 * box[3] + cosTheta2 * box[2];
    pts[1].y = box[1] - cosTheta2 * box[3] - sinTheta2 * box[2];
    pts[2].x = 2 * box[0] - pts[0].x;
    pts[2].y = 2 * box[1] - pts[0].y;
    pts[3].x = 2 * box[0] - pts[1].x;
    pts[3].y = 2 * box[1] - pts[1].y;
  } else if (nms_type == Poly) {
    pts[0].x = box[0];
    pts[0].y = box[1];
    pts[1].x = box[2];
    pts[1].y = box[3];
    pts[2].x = box[4];
    pts[2].y = box[5];
    pts[3].x = box[6];
    pts[3].y = box[7];
  }
}

template <typename T>
T single_box_iou_rotated(T const * const box1_raw, T const * const box2_raw, const int & nms_type=OBB) {
    Point<T> pts1[4];
    Point<T> pts2[4];
    get_rotated_vertices<T>(box1_raw, pts1, nms_type);
    get_rotated_vertices<T>(box2_raw, pts2, nms_type);
    T area1{0}; T area2{0};
    if (nms_type == OBB) {
      area1 = box1_raw[2] * box1_raw[3];
      area2 = box2_raw[2] * box2_raw[3];
    } else if (nms_type == Poly) {
      area1 = polygon_area<T>(box1_raw, 4);
      area2 = polygon_area<T>(box2_raw, 4);
    }
    if (area1 < 1e-10 || area2 < 1e-10) {
        return 0.f;
    }

    T intersection = rotated_boxes_intersection<T>(pts1, pts2, nms_type);
    T iou = intersection / (area1 + area2 - intersection);
    return iou;
}
