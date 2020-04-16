This code is from:
https://bartipan.net/vmath/

things to look at:
-attention: the at(int x, int y) function for matrices is implemented in such a way that x refers 
 to the column and y to the row - this is different from conventional matrix addressing - so any 
 matrix code in this library should be used with transposed indices
 -the () operator, on the other hand, uses regular matrix notation
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
 
 
maybe also look at this:
http://glm.g-truc.net/0.9.5/index.html
because here:
https://stackoverflow.com/questions/17250146/what-and-where-is-vmath-h
someone says, vmath.h is buggy