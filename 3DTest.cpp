#define OLC_PGE_APPLICATION

#include <iostream>
#include <vector>
#include <math.h> 
// #include "olcConsoleGameEngine.h"
#include "olcPixelGameEngine.h"
#include <Windows.h>
#include <algorithm>
#include <string>

struct vec3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct triangle {
    vec3D vertices[3];
};

struct mesh {
    std::vector<triangle> triangles;

    bool LoadFromObjectFile(std::string sFilename) {
        std::ifstream file(sFilename);
        if (!file.is_open()) {
            std::cout << "Unable to load file: " << sFilename << std::endl;
            return false;
        }
        std::cout << sFilename << " loaded" << std::endl;

        // store the unassigned vertices
        std::vector<vec3D> vertices;

        // iterate as long as not at end of file
        while (!file.eof()) {
            char fileline[128];
            file.getline(fileline, 128);

            std::stringstream stream;
            // read current file line
            stream << fileline;

            char junk;

            // get all of the vertex points in the obj text file
            if (fileline[0] == 'v') {
                vec3D vertex;
                // write each "element"(?) of the stringstream into each location
                stream >> junk >> vertex.x >> vertex.y >> vertex.z;
                vertices.push_back(vertex);
            }

            // assign those vertex points to triangles in the obj text file
            if (fileline[0] == 'f') {
                int f[3];
                stream >> junk >> f[0] >> f[1] >> f[2];
                // indices in object file start at 1
                triangles.push_back( { vertices[f[0] - 1], vertices[f[1] - 1], vertices[f[2] - 1] } );
            }
            
        }

        return true;
    }
};

struct matrix4x4 {
    float matrix[4][4] = { 0 };
};

class olc3DEngine : public olc::PixelGameEngine {
public:
    olc3DEngine()
    {
        sAppName = "3D Test";
    }

public:

    // overrides the virtuals of the OG class in the header
    bool OnUserCreate() override
    {

        meshCube.LoadFromObjectFile("axis.obj");

        projectionMatrix = Matrix_MakeProjection(90.0f, (float)ScreenHeight() / (float)ScreenWidth(), 0.1f, 1000.0f);
        for (auto i : projectionMatrix.matrix) {
            std::cout << i[0] << " " << i[1] << " " << i[2] << " " << i[3] << " " << std::endl;
        }
        std::cout << std::endl;
        
        // af, 0, 0, 0
        // 0,  f, 0, 0
        // 0,  0, -q, 1
        // 0,  0, -znearq, 0
        
        return true;
    }

    bool OnUserUpdate(float fElapsedTime) override
    {
        Clear(olc::BLACK);

        if (GetKey(olc::SPACE).bHeld) vCamera.y += 5.0f * fElapsedTime;
        if (GetKey(olc::CTRL).bHeld) vCamera.y -= 5.0f * fElapsedTime;
        if (GetKey(olc::A).bHeld) vCamera.x -= 5.0f * fElapsedTime;
        if (GetKey(olc::D).bHeld) vCamera.x += 5.0f * fElapsedTime;
        if (GetKey(olc::W).bHeld) vCamera.z += 5.0f * fElapsedTime;
        if (GetKey(olc::S).bHeld) vCamera.z -= 5.0f * fElapsedTime;

        vec3D vForward = Vector_Mul(vLookDir, 8.0f * fElapsedTime);

        if (GetKey(olc::RIGHT).bHeld) fYaw -= 2.0f * fElapsedTime;
        if (GetKey(olc::LEFT).bHeld) fYaw += 2.0f * fElapsedTime;
        std::cout << vCamera.x << " " << vCamera.y << " " << vCamera.z << " " << std::endl;
        std::cout << vLookDir.x << " " << vLookDir.y << " " << vLookDir.z << " " << std::endl;

        // 0. reset screen
        // 1. mathematically rotate
        // 2. mathematically translate
        // 3. check if it should be displayed on screen via dot product of normalk
        // 4. project 3d coords into 2d space
        // 5. visually scale it to the screen
        // 6. visually draw

        // rotation matrices
        // fTheta += 1.0f * fElapsedTime;
        matrix4x4 matRotZ = Matrix_MakeRotationZ(fTheta * 0.5);
        matrix4x4 matRotX = Matrix_MakeRotationX(fTheta);

        // translation matrix
        matrix4x4 matTrans = Matrix_MakeTranslation(0.0f, 0.0f, 5.0f);

        // matrix that handles both rotations and translations before projection that will be multiplied with vector
        matrix4x4 matWorldTransformations = Matrix_MakeIdentity();
        matWorldTransformations = Matrix_MultiplyMatrix(matRotZ, matRotX);
        matWorldTransformations = Matrix_MultiplyMatrix(matWorldTransformations, matTrans);

        std::vector<triangle> trianglesToRaster;

        // orienting vector
        vec3D vUp = { 0.0f, 1.0f, 0.0f };
        // point to look at
        vec3D vTarget = { 0.0f, 0.0f, 1.0f };

        // make rotation matrix based on yaw and rotate target to that yaw
        matrix4x4 matCameraRot = Matrix_MakeRotationY(fYaw);
        vLookDir = Matrix_MultiplyVector(matCameraRot, vTarget);

        // unit vector of camera position + direction
        vTarget = Vector_Add(vCamera, vLookDir);

        // current camera location, target point the camera should look at, up direction for orientation
        matrix4x4 matCamera = Matrix_PointAt(vCamera, vTarget, vUp);

        matrix4x4 matView = Matrix_QuickInverse(matCamera);

        // transform and project triangles to see if they are fit to raster
        for (auto tri : meshCube.triangles) {
            triangle triTransformed, triProjected, triViewed;
             
            // transform vertices in triangle
            triTransformed.vertices[0] = Matrix_MultiplyVector(matWorldTransformations, tri.vertices[0]);
            triTransformed.vertices[1] = Matrix_MultiplyVector(matWorldTransformations, tri.vertices[1]);
            triTransformed.vertices[2] = Matrix_MultiplyVector(matWorldTransformations, tri.vertices[2]);

            // normalize current triangle for z-axis to determine if the triangle should be displayed
            vec3D normal, line1, line2;

            line1 = Vector_Sub(triTransformed.vertices[1], triTransformed.vertices[0]);
            line2 = Vector_Sub(triTransformed.vertices[2], triTransformed.vertices[0]);

            normal = Vector_CrossProduct(line1, line2);
            normal = Vector_Normalise(normal);

            vec3D vCameraRay = Vector_Sub(triTransformed.vertices[0], vCamera);

            // dot product = AxBx + AyBy + AzBz
            // used to make comparisons between points
            // used in this program to check if triangle should be displayed
            if (Vector_DotProduct(normal, vCameraRay) < 0.0f) {
                triViewed.vertices[0] = Matrix_MultiplyVector(matView, triTransformed.vertices[0]);
                triViewed.vertices[1] = Matrix_MultiplyVector(matView, triTransformed.vertices[1]);
                triViewed.vertices[2] = Matrix_MultiplyVector(matView, triTransformed.vertices[2]);

                // implement those triangle clipping algorithms to current triangle that passed dot product check
                // Clip Viewed Triangle against near plane, this could form two additional
                // additional triangles. 
                // only for those that are two close (znear); does not handle outside of screen
                int nClippedTriangles = 0;
                triangle clipped[2];
                nClippedTriangles = Triangle_ClipAgainstPlane({ 0.0f, 0.0f, 0.1f }, { 0.0f, 0.0f, 1.0f }, triViewed, clipped[0], clipped[1]);

                for (int n = 0; n < nClippedTriangles; n++) {
                    // project 3d coordinates into 2d space
                    triProjected.vertices[0] = Matrix_MultiplyVector(projectionMatrix, clipped[n].vertices[0]);
                    triProjected.vertices[1] = Matrix_MultiplyVector(projectionMatrix, clipped[n].vertices[1]);
                    triProjected.vertices[2] = Matrix_MultiplyVector(projectionMatrix, clipped[n].vertices[2]);

                    // scale into view via normalizing
                    triProjected.vertices[0] = Vector_Div(triProjected.vertices[0], triProjected.vertices[0].w);
                    triProjected.vertices[1] = Vector_Div(triProjected.vertices[1], triProjected.vertices[1].w);
                    triProjected.vertices[2] = Vector_Div(triProjected.vertices[2], triProjected.vertices[2].w);

                    // increment to offset for final rasterization
                    vec3D vOffsetView = { 1.0f, 1.0f, 0.0f };
                    triProjected.vertices[0] = Vector_Add(triProjected.vertices[0], vOffsetView);
                    triProjected.vertices[1] = Vector_Add(triProjected.vertices[1], vOffsetView);
                    triProjected.vertices[2] = Vector_Add(triProjected.vertices[2], vOffsetView);

                    // scale to half of the screen width
                    triProjected.vertices[0].x *= 0.5f * (float)ScreenWidth();
                    triProjected.vertices[0].y *= 0.5f * (float)ScreenHeight();
                    triProjected.vertices[1].x *= 0.5f * (float)ScreenWidth();
                    triProjected.vertices[1].y *= 0.5f * (float)ScreenHeight();
                    triProjected.vertices[2].x *= 0.5f * (float)ScreenWidth();
                    triProjected.vertices[2].y *= 0.5f * (float)ScreenHeight();

                    trianglesToRaster.push_back(triProjected);
                }
            }
        }

        // sort by providing a condition in a lambda function beginning and ending parameters
        // third parameter tells algorithm to sort in ascending order based on z values
        // not completely perfect (based on average z, inaccuracy errors)
        std::sort(trianglesToRaster.begin(), trianglesToRaster.end(), [](triangle& t1, triangle& t2)
            {
                float avgZ1 = (t1.vertices[0].z + t1.vertices[1].z + t1.vertices[2].z) / 3.0f;
                float avgZ2 = (t2.vertices[0].z + t2.vertices[1].z + t2.vertices[2].z) / 3.0f;
                return avgZ1 > avgZ2;
            }
        );

        // Loop through all transformed, viewed, projected, and sorted triangles
        for (auto& triToRaster : trianglesToRaster)
        {
            // Clip triangles against all four screen edges, this could yield
            // a bunch of triangles, so create a queue that we traverse to 
            //  ensure we only test new triangles generated against planes
            triangle clipped[2];
            std::list<triangle> listTriangles;

            // Add initial triangle
            listTriangles.push_back(triToRaster);
            int nNewTriangles = 1;

            // clip against each screen edge
            for (int p = 0; p < 4; p++)
            {
                int nTrisToAdd = 0;
                while (nNewTriangles > 0)
                {
                    // Take triangle from front of queue
                    triangle test = listTriangles.front();
                    listTriangles.pop_front();
                    nNewTriangles--;

                    // Clip it against a plane. We only need to test each 
                    // subsequent plane, against subsequent new triangles
                    // as all triangles after a plane clip are guaranteed
                    // to lie on the inside of the plane. I like how this
                    // comment is almost completely and utterly justified
                    switch (p)
                    {
                    case 0:	nTrisToAdd = Triangle_ClipAgainstPlane({ 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, test, clipped[0], clipped[1]); break;
                    case 1:	nTrisToAdd = Triangle_ClipAgainstPlane({ 0.0f, (float)ScreenHeight() - 1, 0.0f }, { 0.0f, -1.0f, 0.0f }, test, clipped[0], clipped[1]); break;
                    case 2:	nTrisToAdd = Triangle_ClipAgainstPlane({ 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, test, clipped[0], clipped[1]); break;
                    case 3:	nTrisToAdd = Triangle_ClipAgainstPlane({ (float)ScreenWidth() - 1, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, test, clipped[0], clipped[1]); break;
                    }

                    // Clipping may yield a variable number of triangles, so
                    // add these new ones to the back of the queue for subsequent
                    // clipping against next planes
                    for (int w = 0; w < nTrisToAdd; w++)
                        listTriangles.push_back(clipped[w]);
                }
                nNewTriangles = listTriangles.size();
            }

            // Draw the transformed, viewed, clipped, projected, sorted, clipped triangles
            for (auto& t : listTriangles)
            {
                FillTriangle(t.vertices[0].x, t.vertices[0].y, t.vertices[1].x, t.vertices[1].y, t.vertices[2].x, t.vertices[2].y, olc::WHITE);
                DrawTriangle(t.vertices[0].x, t.vertices[0].y, t.vertices[1].x, t.vertices[1].y, t.vertices[2].x, t.vertices[2].y, olc::BLACK);
            }
        }


        //// rasterize triangles after sorting (painter's algo)
        //for (auto &tri : trianglesToRaster) {
        //    FillTriangle(tri.vertices[0].x, tri.vertices[0].y,
        //        tri.vertices[1].x, tri.vertices[1].y,
        //        tri.vertices[2].x, tri.vertices[2].y,
        //        olc::WHITE
        //    );
        //    DrawTriangle(tri.vertices[0].x, tri.vertices[0].y,
        //        tri.vertices[1].x, tri.vertices[1].y,
        //        tri.vertices[2].x, tri.vertices[2].y,
        //        olc::DARK_GREY
        //    );
        //}

        return true;
    }

    bool OnUserDestroy() override {
        return true;
    }

private:
    // note: all shapes will be comprised of meshes of triangles (including this cube)
    mesh meshCube;
    matrix4x4 projectionMatrix;

    // camera positioning
    vec3D vCamera;
    vec3D vLookDir;

    // camera rotating;
    float fYaw = 0.0f;

    float fTheta = 0.0f;

    // matrix multiplication
    vec3D Matrix_MultiplyVector(matrix4x4& m, vec3D& i) {
        vec3D o;
        o.x = (i.x * m.matrix[0][0]) + (i.y * m.matrix[1][0]) + (i.z * m.matrix[2][0]) + (i.w * m.matrix[3][0]);
        o.y = (i.x * m.matrix[0][1]) + (i.y * m.matrix[1][1]) + (i.z * m.matrix[2][1]) + (i.w * m.matrix[3][1]);
        o.z = (i.x * m.matrix[0][2]) + (i.y * m.matrix[1][2]) + (i.z * m.matrix[2][2]) + (i.w * m.matrix[3][2]);
        o.w = (i.x * m.matrix[0][3]) + (i.y * m.matrix[1][3]) + (i.z * m.matrix[2][3]) + (i.w * m.matrix[3][3]);

        return o;
    }

    // position of object (where it should be), "forward vector", up vector
    matrix4x4 Matrix_PointAt(vec3D& pos, vec3D& target, vec3D& up) {
        // calculate new forward direction (C)
        vec3D newForward = Vector_Sub(target, pos);
        newForward = Vector_Normalise(newForward);

        // calculate new up direction (B)
        vec3D a = Vector_Mul(newForward, Vector_DotProduct(up, newForward));
        // DEBUG: messing around with this rn because y is upside down (x axis points to the negative direction, so its good)
        vec3D newUp = Vector_Sub(a, up);
        newUp = Vector_Normalise(newUp);

        // calculate new right direction (A)
        // uses cross product because x axis is technically normal to y and z axis
        vec3D newRight = Vector_CrossProduct(newForward, newUp);

        matrix4x4 pointAtMatrix;
        pointAtMatrix.matrix[0][0] = newRight.x;    pointAtMatrix.matrix[0][1] = newRight.y;    pointAtMatrix.matrix[0][2] = newRight.z;    pointAtMatrix.matrix[0][3] = 0.0f;
        pointAtMatrix.matrix[1][0] = newUp.x;       pointAtMatrix.matrix[1][1] = newUp.y;       pointAtMatrix.matrix[1][2] = newUp.z;       pointAtMatrix.matrix[1][3] = 0.0f;
        pointAtMatrix.matrix[2][0] = newForward.x;  pointAtMatrix.matrix[2][1] = newForward.y;  pointAtMatrix.matrix[2][2] = newForward.z;  pointAtMatrix.matrix[2][3] = 0.0f;
        pointAtMatrix.matrix[3][0] = pos.x;         pointAtMatrix.matrix[3][1] = pos.y;         pointAtMatrix.matrix[3][2] = pos.z;         pointAtMatrix.matrix[3][3] = 1.0f;
        
        return pointAtMatrix;
    }

    matrix4x4 Matrix_QuickInverse(matrix4x4& m) // Only for Rotation/Translation Matrices
    {
        matrix4x4 matrix;
        matrix.matrix[0][0] = m.matrix[0][0]; matrix.matrix[0][1] = m.matrix[1][0]; matrix.matrix[0][2] = m.matrix[2][0]; matrix.matrix[0][3] = 0.0f;
        matrix.matrix[1][0] = m.matrix[0][1]; matrix.matrix[1][1] = m.matrix[1][1]; matrix.matrix[1][2] = m.matrix[2][1]; matrix.matrix[1][3] = 0.0f;
        matrix.matrix[2][0] = m.matrix[0][2]; matrix.matrix[2][1] = m.matrix[1][2]; matrix.matrix[2][2] = m.matrix[2][2]; matrix.matrix[2][3] = 0.0f;
        matrix.matrix[3][0] = -(m.matrix[3][0] * matrix.matrix[0][0] + m.matrix[3][1] * matrix.matrix[1][0] + m.matrix[3][2] * matrix.matrix[2][0]);
        matrix.matrix[3][1] = -(m.matrix[3][0] * matrix.matrix[0][1] + m.matrix[3][1] * matrix.matrix[1][1] + m.matrix[3][2] * matrix.matrix[2][1]);
        matrix.matrix[3][2] = -(m.matrix[3][0] * matrix.matrix[0][2] + m.matrix[3][1] * matrix.matrix[1][2] + m.matrix[3][2] * matrix.matrix[2][2]);
        matrix.matrix[3][3] = 1.0f;
        return matrix;
    }

    matrix4x4 Matrix_MakeIdentity()
    {
        matrix4x4 matrix;
        matrix.matrix[0][0] = 1.0f;
        matrix.matrix[1][1] = 1.0f;
        matrix.matrix[2][2] = 1.0f;
        matrix.matrix[3][3] = 1.0f;
        return matrix;
    }

    matrix4x4 Matrix_MakeRotationX(float fAngleRad)
    {
        matrix4x4 matrix;
        matrix.matrix[0][0] = 1.0f;
        matrix.matrix[1][1] = cosf(fAngleRad);
        matrix.matrix[1][2] = sinf(fAngleRad);
        matrix.matrix[2][1] = -sinf(fAngleRad);
        matrix.matrix[2][2] = cosf(fAngleRad);
        matrix.matrix[3][3] = 1.0f;
        return matrix;
    }

    matrix4x4 Matrix_MakeRotationY(float fAngleRad)
    {
        matrix4x4 matrix;
        matrix.matrix[0][0] = cosf(fAngleRad);
        matrix.matrix[0][2] = sinf(fAngleRad);
        matrix.matrix[2][0] = -sinf(fAngleRad);
        matrix.matrix[1][1] = 1.0f;
        matrix.matrix[2][2] = cosf(fAngleRad);
        matrix.matrix[3][3] = 1.0f;
        return matrix;
    }

    matrix4x4 Matrix_MakeRotationZ(float fAngleRad)
    {
        matrix4x4 matrix;
        matrix.matrix[0][0] = cosf(fAngleRad);
        matrix.matrix[0][1] = sinf(fAngleRad);
        matrix.matrix[1][0] = -sinf(fAngleRad);
        matrix.matrix[1][1] = cosf(fAngleRad);
        matrix.matrix[2][2] = 1.0f;
        matrix.matrix[3][3] = 1.0f;
        return matrix;
    }

    matrix4x4 Matrix_MakeTranslation(float x, float y, float z)
    {
        matrix4x4 matrix;
        matrix.matrix[0][0] = 1.0f;
        matrix.matrix[1][1] = 1.0f;
        matrix.matrix[2][2] = 1.0f;
        matrix.matrix[3][3] = 1.0f;
        matrix.matrix[3][0] = x;
        matrix.matrix[3][1] = y;
        matrix.matrix[3][2] = z;
        return matrix;
    }

    matrix4x4 Matrix_MakeProjection(float fFovDegrees, float fAspectRatio, float fNear, float fFar)
    {
        float fFovRad = 1.0f / tanf(fFovDegrees * 0.5f / 180.0f * 3.14159f);
        matrix4x4 matrix;
        matrix.matrix[0][0] = fAspectRatio * fFovRad;
        matrix.matrix[1][1] = fFovRad;
        matrix.matrix[2][2] = fFar / (fFar - fNear);
        matrix.matrix[3][2] = (-fFar * fNear) / (fFar - fNear);
        matrix.matrix[2][3] = 1.0f;
        matrix.matrix[3][3] = 0.0f;
        return matrix;
    }

    matrix4x4 Matrix_MultiplyMatrix(matrix4x4& m1, matrix4x4& m2)
    {
        matrix4x4 matrix;
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                matrix.matrix[r][c] = m1.matrix[r][0] * m2.matrix[0][c] + m1.matrix[r][1] * m2.matrix[1][c] + m1.matrix[r][2] * m2.matrix[2][c] + m1.matrix[r][3] * m2.matrix[3][c];
        return matrix;
    }

    vec3D Vector_Add(vec3D& v1, vec3D& v2)
    {
        return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
    }

    vec3D Vector_Sub(vec3D& v1, vec3D& v2)
    {
        return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
    }

    vec3D Vector_Mul(vec3D& v1, float k)
    {
        return { v1.x * k, v1.y * k, v1.z * k };
    }

    vec3D Vector_Div(vec3D& v1, float k)
    {
        if (k == 0.0f) return { 0.0f, 0.0f, 0.0f };
        return { v1.x / k, v1.y / k, v1.z / k };
    }

    float Vector_DotProduct(vec3D& v1, vec3D& v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    float Vector_Length(vec3D& v)
    {
        return sqrtf(Vector_DotProduct(v, v));
    }

    vec3D Vector_Normalise(vec3D& v)
    {
        float l = Vector_Length(v);
        return { v.x / l, v.y / l, v.z / l };
    }

    vec3D Vector_CrossProduct(vec3D& v1, vec3D& v2)
    {
        vec3D v;
        v.x = v1.y * v2.z - v1.z * v2.y;
        v.y = v1.z * v2.x - v1.x * v2.z;
        v.z = v1.x * v2.y - v1.y * v2.x;
        return v;
    }

    // plane_p = plane's point, plane_n = plane's normal
    // return the point P where a line intersects with a plane
    // NxPx + NyPy + NzPz - P * N = 0
    vec3D Vector_IntersectPlane(vec3D& plane_p, vec3D& plane_n, vec3D& lineStart, vec3D& lineEnd)
    {
        plane_n = Vector_Normalise(plane_n);
        float plane_d = -Vector_DotProduct(plane_n, plane_p);
        float ad = Vector_DotProduct(lineStart, plane_n);
        float bd = Vector_DotProduct(lineEnd, plane_n);
        float t = (-plane_d - ad) / (bd - ad);
        vec3D lineStartToEnd = Vector_Sub(lineEnd, lineStart);
        vec3D lineToIntersect = Vector_Mul(lineStartToEnd, t);
        return Vector_Add(lineStart, lineToIntersect);
    }

    // returns how many triangles will be produced
    // note: signed distance (displacement) is used to determine if triangle needs clipping first (and how much),
    //       then Vector_IntersectPlane is used to find where that triangle should be clipped in world space
    int Triangle_ClipAgainstPlane(vec3D plane_p, vec3D plane_n, triangle& inputTri, triangle& outputTri1, triangle& outputTri2)
    {
        // make sure plane normal is normal
        plane_n = Vector_Normalise(plane_n);

        auto distance = [&](vec3D& point)
        {
            vec3D n = Vector_Normalise(point);
            return (plane_n.x * point.x + plane_n.y * point.y + plane_n.z * point.z - Vector_DotProduct(plane_n, plane_p));
        };

        // Create two temporary storage arrays to classify points either side of plane
        // If distance sign is positive, point lies on "inside" of plane
        vec3D* inside_points[3];  int nInsidePointCount = 0;
        vec3D* outside_points[3]; int nOutsidePointCount = 0;

        // Get signed distance of each point in triangle to plane
        float d0 = distance(inputTri.vertices[0]);
        float d1 = distance(inputTri.vertices[1]);
        float d2 = distance(inputTri.vertices[2]);

        // if distance is positive, then point is inside plane
        if (d0 >= 0) { inside_points[nInsidePointCount++] = &inputTri.vertices[0]; }
        else { outside_points[nOutsidePointCount++] = &inputTri.vertices[0]; }
        if (d1 >= 0) { inside_points[nInsidePointCount++] = &inputTri.vertices[1]; }
        else { outside_points[nOutsidePointCount++] = &inputTri.vertices[1]; }
        if (d2 >= 0) { inside_points[nInsidePointCount++] = &inputTri.vertices[2]; }
        else { outside_points[nOutsidePointCount++] = &inputTri.vertices[2]; }

        if (nInsidePointCount == 0) {
            return 0;
        }

        if (nInsidePointCount == 3) {
            outputTri1 = inputTri;
            return 1;
        }

        if (nInsidePointCount == 1 && nOutsidePointCount == 2) {
            // Triangle should be clipped. As two points lie outside
            // the plane, the triangle simply becomes a smaller triangle

            // The inside point is valid, so keep that...
            outputTri1.vertices[0] = *inside_points[0];

            // but the two new points are at the locations where the 
            // original sides of the triangle (lines) intersect with the plane
            outputTri1.vertices[1] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[0]);
            outputTri1.vertices[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[1]);

            return 1; // Return the newly formed single triangle
        }

        if (nInsidePointCount == 2 && nOutsidePointCount == 1) {
            // The first triangle consists of the two inside points and a new
            // point determined by the location where one side of the triangle
            // intersects with the plane
            outputTri1.vertices[0] = *inside_points[0];
            outputTri1.vertices[1] = *inside_points[1];
            outputTri1.vertices[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[0]);

            // The second triangle is composed of one of he inside points, a
            // new point determined by the intersection of the other side of the 
            // triangle and the plane, and the newly created point above
            outputTri2.vertices[0] = *inside_points[1];
            outputTri2.vertices[1] = outputTri1.vertices[2];
            outputTri2.vertices[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[1], *outside_points[0]);
            
            return 2;
        }

    }
};

int main(int argc, char **argv[])
{
    olc3DEngine test3D;
    if (test3D.Construct(400, 300, 2, 2)) {
        test3D.Start();
    }
    return 0;
}


