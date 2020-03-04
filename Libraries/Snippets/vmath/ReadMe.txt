This code is from:
https://bartipan.net/vmath/

things to look at:
-how unions are used in Vector3 for having (x,y,z),(r,g,b),(s,t,u) vectors
-Vector3::rotate - compare to my implementation
-Matrix3::createRotationAroundAxis
-Matrix3::inverse() - seems like a clean and efficient formula
-Matrix4::createRotationAroundAxis, createTranslation, createScale
-Projection:
 -Matrix4::createLookAt, createFrustum, createOrtho
-Matrix4::det
-Matrix4::inverse
-Quaternion:
 -how it uses a sclar for the "real" part and a 3-vector for the "imaginary"
 -operator* - quaternion multiplication
 -length
 -fromEulerAngles, fromAxisRot, fromMatrix (2 versions)
 -rotMatrix, transform
 -slerp (spherical interpolation)
-Aabb: axis-aligned bounding box:
 -intersects (2 versions), intersection