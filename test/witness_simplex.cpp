/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, Jelle Feringa
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/// @file witness_simplex.cpp
/// @brief A contact's and a distance result's witness simplex: the weights of
///        the final GJK / EPA simplex and the support hints of its vertices,
///        per shape and in the result's object order.

#define BOOST_TEST_MODULE COAL_WITNESS_SIMPLEX
#include <boost/test/included/unit_test.hpp>

#include "coal/collision.h"
#include "coal/distance.h"
#include "coal/BVH/BVH_model.h"
#include "coal/math/transform.h"
#include "coal/shape/geometric_shapes.h"

#ifdef COAL_HAS_OCTOMAP
#include "coal/octree.h"
#endif

using namespace coal;

namespace {
// The segment from A to B, as a convex polytope with two vertices: A is vertex
// 0, B vertex 1. Its support function writes the index of the vertex it
// returned into the hint; ties go to A.
const Vec3s SEGMENT_A(-1, 0, 0);
const Vec3s SEGMENT_B(3, 0, 0);
// Every bench below puts the one nearest point of the segment at the origin, a
// quarter of the way from A to B: the weight on B's vertices is 1/4.
constexpr Scalar WEIGHT_ON_B = 0.25;
// GJK and EPA end on exact polytope features here; the weights are exact up to
// double round-off on unit-size coordinates.
constexpr Scalar WEIGHT_TOL = 1e-9;

class Segment : public ConvexBase32 {
 public:
  Segment() {
    this->initialize(std::make_shared<std::vector<Vec3s>>(
                         std::vector<Vec3s>{SEGMENT_A, SEGMENT_B}),
                     2);
  }
};

std::shared_ptr<Segment> makeSegment() {
  auto segment = std::make_shared<Segment>();
  segment->computeLocalAABB();
  return segment;
}

// A cube with its body diagonal along z, centred 2 above the origin: its one
// lowest point, a vertex, is straight above the origin.
std::shared_ptr<Box> makeCube() {
  auto cube = std::make_shared<Box>(0.5, 0.5, 0.5);
  cube->computeLocalAABB();
  return cube;
}
Transform3s cubePose() {
  const Matrix3s turn = Eigen::Quaternion<Scalar>::FromTwoVectors(
                            Vec3s(1, 1, 1).normalized(), Vec3s(0, 0, 1))
                            .toRotationMatrix();
  return Transform3s(turn, Vec3s(0, 0, 2));
}

// One triangle whose lowest point, a vertex, is 1.5 straight above the origin:
// every other point is higher, so farther from the segment's line.
std::shared_ptr<BVHModel<OBBRSS>> makeTriangleMesh() {
  auto mesh = std::make_shared<BVHModel<OBBRSS>>();
  std::vector<Vec3s> points{Vec3s(0, 0, 1.5), Vec3s(-0.3, -0.2, 2.5),
                            Vec3s(0.3, 0.2, 2.6)};
  std::vector<Triangle32> triangles{Triangle32(0, 1, 2)};
  mesh->beginModel();
  mesh->addSubModel(points, triangles);
  mesh->endModel();
  return mesh;
}

// The segment's nearest point: the combination of A and B the witness simplex
// names.
Vec3s segmentPoint(Scalar weight_on_b) {
  return (1 - weight_on_b) * SEGMENT_A + weight_on_b * SEGMENT_B;
}

// A rigid motion of the whole scene: the weights do not move, the witness
// simplex's support points (world frame) move with it.
Transform3s sceneMotion() {
  const Matrix3s turn =
      Eigen::AngleAxis<Scalar>(0.7, Vec3s(1, -2, 0.5).normalized())
          .toRotationMatrix();
  return Transform3s(turn, Vec3s(0.4, -1.3, 2.2));
}

void checkWeights(const WitnessSimplex& witness, int segment_side) {
  BOOST_REQUIRE_GE(witness.rank, 1);
  BOOST_REQUIRE_LE(witness.rank, 4);
  Scalar total = 0;
  for (int k = 0; k < witness.rank; ++k)
    total += witness.weights[static_cast<std::size_t>(k)];
  BOOST_CHECK_CLOSE_FRACTION(total, 1, WEIGHT_TOL);
  BOOST_CHECK_SMALL(witness.weightOnSupportHint(segment_side, 1) - WEIGHT_ON_B,
                    WEIGHT_TOL);
  BOOST_CHECK_SMALL(
      witness.weightOnSupportHint(segment_side, 0) - (1 - WEIGHT_ON_B),
      WEIGHT_TOL);
}

// The support points: in the world frame, the segment's are its end the hint
// names, and on each side the weights combine them into the witness point.
void checkSupportPoints(const WitnessSimplex& witness, int segment_side,
                        const Transform3s& scene,
                        const std::array<Vec3s, 2>& nearest_points) {
  BOOST_REQUIRE_GE(witness.rank, 1);
  for (int k = 0; k < witness.rank; ++k) {
    const std::size_t kk = static_cast<std::size_t>(k);
    const Vec3s end = scene.transform(
        witness.hints[kk][segment_side] == 0 ? SEGMENT_A : SEGMENT_B);
    BOOST_CHECK_SMALL(
        (witness.support_points[static_cast<std::size_t>(segment_side)][kk] -
         end)
            .norm(),
        WEIGHT_TOL);
  }
  for (std::size_t side = 0; side < 2; ++side) {
    Vec3s combined = Vec3s::Zero();
    for (int k = 0; k < witness.rank; ++k)
      combined += witness.weights[static_cast<std::size_t>(k)] *
                  witness.support_points[side][static_cast<std::size_t>(k)];
    BOOST_CHECK_SMALL((combined - nearest_points[side]).norm(), WEIGHT_TOL);
  }
}
}  // namespace

BOOST_AUTO_TEST_CASE(shape_shape_distance_and_collide) {
  const auto segment = makeSegment();
  const auto cube = makeCube();

  for (const Transform3s& id : {Transform3s(), sceneMotion()})
    for (const bool segment_first : {true, false}) {
      const int side = segment_first ? 0 : 1;
      const std::size_t ns = static_cast<std::size_t>(side);
      const Transform3s cube_pose = id * cubePose();
      DistanceRequest dreq;
      DistanceResult dres;
      if (segment_first)
        distance(segment.get(), id, cube.get(), cube_pose, dreq, dres);
      else
        distance(cube.get(), cube_pose, segment.get(), id, dreq, dres);
      checkWeights(dres.witness, side);
      checkSupportPoints(dres.witness, side, id, dres.nearest_points);
      BOOST_CHECK_SMALL(
          (dres.nearest_points[ns] - id.transform(Vec3s::Zero())).norm(),
          WEIGHT_TOL);

      CollisionRequest creq;
      creq.security_margin = 2;
      CollisionResult cres;
      if (segment_first)
        collide(segment.get(), id, cube.get(), cube_pose, creq, cres);
      else
        collide(cube.get(), cube_pose, segment.get(), id, creq, cres);
      BOOST_REQUIRE_EQUAL(cres.numContacts(), 1);
      checkWeights(cres.getContact(0).witness, side);
      checkSupportPoints(cres.getContact(0).witness, side, id,
                         cres.getContact(0).nearest_points);
    }
}

BOOST_AUTO_TEST_CASE(mesh_shape_distance_and_collide) {
  const auto segment = makeSegment();
  const auto mesh = makeTriangleMesh();

  for (const Transform3s& id : {Transform3s(), sceneMotion()})
    for (const bool segment_first : {true, false}) {
      const int side = segment_first ? 0 : 1;
      const std::size_t ns = static_cast<std::size_t>(side);
      DistanceRequest dreq;
      DistanceResult dres;
      if (segment_first)
        distance(segment.get(), id, mesh.get(), id, dreq, dres);
      else
        distance(mesh.get(), id, segment.get(), id, dreq, dres);
      checkWeights(dres.witness, side);
      checkSupportPoints(dres.witness, side, id, dres.nearest_points);
      BOOST_CHECK_SMALL(
          (dres.nearest_points[ns] - id.transform(Vec3s::Zero())).norm(),
          WEIGHT_TOL);

      CollisionRequest creq;
      creq.security_margin = 2;
      CollisionResult cres;
      if (segment_first)
        collide(segment.get(), id, mesh.get(), id, creq, cres);
      else
        collide(mesh.get(), id, segment.get(), id, creq, cres);
      BOOST_REQUIRE_EQUAL(cres.numContacts(), 1);
      checkWeights(cres.getContact(0).witness, side);
      checkSupportPoints(cres.getContact(0).witness, side, id,
                         cres.getContact(0).nearest_points);
    }
}

BOOST_AUTO_TEST_CASE(penetration_witness_is_the_weighted_combination) {
  // The segment pierces a box along x; the box's bottom is 0.2 below the
  // segment, its other faces farther: EPA's depth is 0.2 along z and its
  // witness on the segment any point with x in [-0.1, 0.1]. Whichever it picks,
  // the witness is the combination of A and B the weights name, at a weight on
  // B in [0.225, 0.275].
  const auto segment = makeSegment();
  auto box = std::make_shared<Box>(0.2, 2, 1);
  box->computeLocalAABB();
  const Transform3s id;
  const Transform3s box_pose(Matrix3s::Identity(), Vec3s(0, 0, 0.3));

  for (const bool segment_first : {true, false}) {
    const int side = segment_first ? 0 : 1;
    const std::size_t ns = static_cast<std::size_t>(side);
    CollisionRequest creq;
    creq.enable_contact = true;
    CollisionResult cres;
    if (segment_first)
      collide(segment.get(), id, box.get(), box_pose, creq, cres);
    else
      collide(box.get(), box_pose, segment.get(), id, creq, cres);
    BOOST_REQUIRE_EQUAL(cres.numContacts(), 1);
    const Contact& c = cres.getContact(0);
    BOOST_CHECK_CLOSE_FRACTION(c.penetration_depth, -0.2, WEIGHT_TOL);
    const Scalar on_b = c.witness.weightOnSupportHint(side, 1);
    BOOST_CHECK_GE(on_b, 0.225 - WEIGHT_TOL);
    BOOST_CHECK_LE(on_b, 0.275 + WEIGHT_TOL);
    BOOST_CHECK_SMALL((c.nearest_points[ns] - segmentPoint(on_b)).norm(),
                      WEIGHT_TOL);
    checkSupportPoints(c.witness, side, id, c.nearest_points);
  }
}

#ifdef COAL_HAS_OCTOMAP
BOOST_AUTO_TEST_CASE(octree_shape_collide) {
  // One voxel, [0, 0.25] x [0, 0.25] x [1, 1.25]: its bottom edge at y = 0 is
  // parallel to the segment, 1 above it, so the witness on the segment is any
  // point with x in [0, 0.25], at a weight on B in [0.25, 0.3125]. Whichever
  // it is, it is the combination of A and B the weights name.
  constexpr Scalar resolution = 0.25;
  auto tree = std::make_shared<octomap::OcTree>(resolution);
  tree->updateNode(octomap::point3d(0.125f, 0.125f, 1.125f), true);
  const OcTree octree(tree);
  const auto segment = makeSegment();
  const Transform3s id;

  for (const bool segment_first : {true, false}) {
    CollisionRequest creq;
    creq.security_margin = 2;
    CollisionResult cres;
    if (segment_first)
      collide(segment.get(), id, &octree, id, creq, cres);
    else
      collide(&octree, id, segment.get(), id, creq, cres);
    BOOST_REQUIRE_EQUAL(cres.numContacts(), 1);
    const Contact& c = cres.getContact(0);
    // A shape-octree collision reports the octree as o1 in either call order;
    // the witness simplex follows the contact's own order.
    const int side = c.o1 == segment.get() ? 0 : 1;
    const std::size_t ns = static_cast<std::size_t>(side);
    BOOST_TEST_MESSAGE("segment_first " << segment_first << ", side " << side);
    BOOST_REQUIRE_GE(c.witness.rank, 1);
    const Scalar on_b = c.witness.weightOnSupportHint(side, 1);
    BOOST_CHECK_GE(on_b, 0.25 - WEIGHT_TOL);
    BOOST_CHECK_LE(on_b, 0.3125 + WEIGHT_TOL);
    BOOST_CHECK_SMALL((c.nearest_points[ns] - segmentPoint(on_b)).norm(),
                      WEIGHT_TOL);
    checkSupportPoints(c.witness, side, id, c.nearest_points);
  }
}
#endif

BOOST_AUTO_TEST_CASE(no_witness_without_a_simplex) {
  // A result that no GJK run filled carries rank 0.
  const DistanceResult dres;
  BOOST_CHECK_EQUAL(dres.witness.rank, 0);
  const Contact c;
  BOOST_CHECK_EQUAL(c.witness.rank, 0);
}
