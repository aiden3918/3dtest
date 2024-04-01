#define OLC_PGE_APPLICATION

#include <iostream>
#include <vector>
#include <math.h> 
// #include "olcConsoleGameEngine.h"
#include "olcPixelGameEngine.h"
#include "coolerMath.h"
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
            std::cout << fileline << std::endl;

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

        std::cout << "All possible vertices: " << std::endl;
        for (auto i : vertices) {
            std::cout << i.x << " " << i.y << " " << i.z << std::endl;
        }

        std::cout << "All triangle coordinates: " << std::endl;
        for (auto i : triangles) {
            std::cout << "{ " << i.vertices[0].x << ", " << i.vertices[0].y << ", " << i.vertices[0].z << " }" << std::endl;
            std::cout << "{ " << i.vertices[1].x << ", " << i.vertices[1].y << ", " << i.vertices[1].z << " }" << std::endl;
            std::cout << "{ " << i.vertices[2].x << ", " << i.vertices[2].y << ", " << i.vertices[2].z << " }" << std::endl << std::endl;
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
        // origin is at the frontmost bottom left of the cube
        /*
            * |\--|
            * | \ |
            * |__\|
        */
        // triangle: {{x, y, z}, {x, y, z}, {x, y, z}}

        meshCube.triangles = {
            // SOUTH
            { 0.0f, 0.0f, 0.0f,    0.0f, 1.0f, 0.0f,    1.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 0.0f,    1.0f, 1.0f, 0.0f,    1.0f, 0.0f, 0.0f },
            // EAST                                                      
            { 1.0f, 0.0f, 0.0f,    1.0f, 1.0f, 0.0f,    1.0f, 1.0f, 1.0f },
            { 1.0f, 0.0f, 0.0f,    1.0f, 1.0f, 1.0f,    1.0f, 0.0f, 1.0f },
            // NORTH                                                     
            { 1.0f, 0.0f, 1.0f,    1.0f, 1.0f, 1.0f,    0.0f, 1.0f, 1.0f },
            { 1.0f, 0.0f, 1.0f,    0.0f, 1.0f, 1.0f,    0.0f, 0.0f, 1.0f },
            // WEST                                                      
            { 0.0f, 0.0f, 1.0f,    0.0f, 1.0f, 1.0f,    0.0f, 1.0f, 0.0f },
            { 0.0f, 0.0f, 1.0f,    0.0f, 1.0f, 0.0f,    0.0f, 0.0f, 0.0f },
            // TOP                                                       
            { 0.0f, 1.0f, 0.0f,    0.0f, 1.0f, 1.0f,    1.0f, 1.0f, 1.0f },
            { 0.0f, 1.0f, 0.0f,    1.0f, 1.0f, 1.0f,    1.0f, 1.0f, 0.0f },
            // BOTTOM                                                    
            { 1.0f, 0.0f, 1.0f,    0.0f, 0.0f, 1.0f,    0.0f, 0.0f, 0.0f },
            { 1.0f, 0.0f, 1.0f,    0.0f, 0.0f, 0.0f,    1.0f, 0.0f, 0.0f },
        };

        // meshCube.LoadFromObjectFile("VideoShip.obj");

        matrix4x4 projectionMatrix = Matrix_MakeProjection(90.0f, (float)ScreenHeight() / (float)ScreenWidth(), 0.1f, 1000.0f);
        for (auto i : projectionMatrix.matrix) {
            std::cout << i[0] << " " << i[1] << " " << i[2] << " " << i[3] << " " << std::endl;
        }
        std::cout << std::endl;


        /*
        af, 0, 0, 0
        0,  f, 0, 0
        0,  0, -q, 1
        0,  0, -znearq, 0
        */

        return true;
    }

    bool OnUserUpdate(float fElapsedTime) override
    {
        Clear(olc::BLACK);

        /*
        * 0. reset screen
        * 1. mathematically rotated
        * 2. mathematically translate
        * 3. check if it should be displayed on screen via its normal
        * 4. project 3d coords into 2d space
        * 5. visually scale it to the screen
        * 6. visually draw
        */

        // rotation matrices
        fTheta += 1.0f * fElapsedTime;
        matrix4x4 matRotZ = Matrix_MakeRotationZ(fTheta * 0.5);
        matrix4x4 matRotX = Matrix_MakeRotationX(fTheta);

        // translation matrix
        matrix4x4 matTrans = Matrix_MakeTranslation(0.0f, 0.0f, 3.0f);

        // matrix that handles both rotations and translations before projection that will be multiplied with vector
        matrix4x4 matWorldTransformations = Matrix_MakeIdentity();
        matWorldTransformations = Matrix_MultiplyMatrix(matRotZ, matRotX);
        matWorldTransformations = Matrix_MultiplyMatrix(matWorldTransformations, matTrans);

        std::vector<triangle> trianglesToRaster;

        // transform and project triangles to see if they are fit to raster
        for (auto tri : meshCube.triangles) {
            triangle triTransformed;

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
            // tbh i dont fully understand it myself but javid made it work so
            if (Vector_DotProduct(normal, vCameraRay) < 0.0f) {
                triangle triProjected;
                /*vec3D lightDirection = { 0.0f, 0.0f, -1.0f };
                float lightMagnitude = sqrtf((lightDirection.x * lightDirection.x) + (lightDirection.y * lightDirection.y) + (lightDirection.z * lightDirection.z));
                lightDirection.x /= lightMagnitude; lightDirection.y /= lightMagnitude; lightDirection.z /= lightMagnitude;

                float dotProdLighting = (lightDirection.x * normal.x) + (lightDirection.y + normal.y) + (lightDirection.z + normal.z);*/

                // project 3d coordinates into 2d space
                // DEBUG: problem seems to arise here, everything in triprojected becomes 0
                for (auto i : triTransformed.vertices) {
                    std::cout << i.x << " " << i.y << " " << i.z << " " << i.w << " " << std::endl;
                }
                std::cout << std::endl;
                triProjected.vertices[0] = Matrix_MultiplyVector(projectionMatrix, triTransformed.vertices[0]);
                triProjected.vertices[1] = Matrix_MultiplyVector(projectionMatrix, triTransformed.vertices[1]);
                triProjected.vertices[2] = Matrix_MultiplyVector(projectionMatrix, triTransformed.vertices[2]);
                for (auto i : triProjected.vertices) {
                    std::cout << i.x << " " << i.y << " " << i.z << " " << i.w << " " << std::endl;
                }
                std::cout << std::endl;

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
                triProjected.vertices[0].x *= 0.5f * (float)ScreenWidth(); triProjected.vertices[0].y *= 0.5f * (float)ScreenHeight();
                triProjected.vertices[1].x *= 0.5f * (float)ScreenWidth(); triProjected.vertices[1].y *= 0.5f * (float)ScreenHeight();
                triProjected.vertices[2].x *= 0.5f * (float)ScreenWidth(); triProjected.vertices[2].y *= 0.5f * (float)ScreenHeight();


                //FillTriangle(triProjected.vertices[0].x, triProjected.vertices[0].y,
                //    triProjected.vertices[1].x, triProjected.vertices[1].y,
                //    triProjected.vertices[2].x, triProjected.vertices[2].y,
                //    olc::DARK_GREY
                //);
                //DrawTriangle(triProjected.vertices[0].x, triProjected.vertices[0].y,
                //    triProjected.vertices[1].x, triProjected.vertices[1].y,
                //    triProjected.vertices[2].x, triProjected.vertices[2].y,
                //    olc::WHITE
                //);

                trianglesToRaster.push_back(triProjected);
            }

            // sort by providing a condition in a lambda function beginning and ending parameters
            // third parameter tells algorithm to sort in ascending order based on z values
            // not completely perfect (based on average z, inaccuracy errors)
            std::sort(trianglesToRaster.begin(), trianglesToRaster.end(), [](triangle& triangle1, triangle& triangle2)
                {
                    float avgZ1 = (triangle1.vertices[0].z + triangle1.vertices[1].z + triangle1.vertices[2].z) / 3.0f;
                    float avgZ2 = (triangle2.vertices[0].z + triangle2.vertices[1].z + triangle2.vertices[2].z) / 3.0f;
                    return avgZ1 > avgZ2;
                }
            );
            // rasterize triangles after sorting (painter's algo)
            for (auto &triangle: trianglesToRaster) {
                FillTriangle(triangle.vertices[0].x, triangle.vertices[0].y,
                    triangle.vertices[1].x, triangle.vertices[1].y,
                    triangle.vertices[2].x, triangle.vertices[2].y,
                    olc::DARK_GREY
                );
                DrawTriangle(triangle.vertices[0].x, triangle.vertices[0].y,
                    triangle.vertices[1].x, triangle.vertices[1].y,
                    triangle.vertices[2].x, triangle.vertices[2].y,
                    olc::WHITE
                );
            }
        }

        return true;
    }

    bool OnUserDestroy() override {
        return true;
    }

private:
    // note: all shapes will be comprised of meshes of triangles (including this cube)
    mesh meshCube;
    matrix4x4 projectionMatrix;

    vec3D vCamera = { 0.0f, 0.0f, 0.0f };

    float fTheta = 0.0f;

    // matrix multiplication
    vec3D Matrix_MultiplyVector(matrix4x4& matrix, vec3D& inputVec) {
        vec3D outputVec;
        outputVec.x = (inputVec.x * matrix.matrix[0][0]) + (inputVec.y * matrix.matrix[1][0]) + (inputVec.z * matrix.matrix[2][0]) + (inputVec.w * matrix.matrix[3][0]);
        outputVec.y = (inputVec.x * matrix.matrix[0][1]) + (inputVec.y * matrix.matrix[1][1]) + (inputVec.z * matrix.matrix[2][1]) + (inputVec.w * matrix.matrix[3][1]);
        outputVec.z = (inputVec.x * matrix.matrix[0][2]) + (inputVec.y * matrix.matrix[1][2]) + (inputVec.z * matrix.matrix[2][2]) + (inputVec.w * matrix.matrix[3][2]);
        outputVec.w = (inputVec.x * matrix.matrix[0][3]) + (inputVec.y * matrix.matrix[1][3]) + (inputVec.z * matrix.matrix[2][3]) + (inputVec.w * matrix.matrix[3][3]);

        return outputVec;
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
};

int main(int argc, char **argv[])
{
    olc3DEngine test3D;
    if (test3D.Construct(400, 300, 2, 2)) {
        test3D.Start();
    }
    return 0;
}
