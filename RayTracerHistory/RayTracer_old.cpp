//
//  RayTracer.cpp
//  
//
//  Built by Ian Chen and Betty Chen at Mar 2013
//
//
//  version Mar5 00:44
//

#include "FreeImage.h"
#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include <vector>
#include <math.h>


#define WIDTH 640
#define HEIGHT 480
#define BPP 24
#define PI 3.1415926535897932384626

using namespace std;
using namespace Eigen;


//***********************************************
//  Utility Function Claim
//***********************************************

Vector4f times(Vector4f, Vector4f);
Vector4f normalize(Vector4f);
Matrix4f coordinate_convertor(Matrix4f, Matrix4f);
Vector4f M_V_Convert(Matrix4f a, Vector4f b);

//***********************************************
//  Classes
//***********************************************
class Color{
    public:
        Vector4f intensity;
        Color (float r, float g, float b){  
            intensity = Vector4f(r, g, b, 0.0);
        }

        Vector4f get_color(){
            return intensity;
        }

        Vector4f set_color(Vector4f co){
            intensity[0] = co[0];
            intensity[1] = co[1];
            intensity[2] = co[2];
        }
};


class Ray{
    public:
        Vector4f origin;        // point (x, y, z, 1)
        Vector4f direction;     // vector (x, y, z, 0)
//        float t;
        // Color intensity;     // rgb value
    
        // Ray (Vector4f camara, Vector4f direc) : intensity(0.0, 0.0, 0.0){
        Ray(){
            //do nothing
        }

        Ray(Vector4f camara, Vector4f direc){
            origin = camara;
            direction = direc;
        }

    
        Vector4f shootpoint(float t){
            return origin + direction * t;
        }
    
        float shootdistance(Vector4f point){
            Vector4f shoot = point - origin;
            float t = (float) shoot[0]/direction[0];
            // if (t < 0){
            //     cout << "\n";
            //     cout << "point is " << point << endl;
            //     cout << "origin is " << origin << endl;
            //     cout << "direction is " << direction << endl;
            // }
            return t;            
        }
    
        // Vector4f set_intensity (Vector4f rgbIntensity){
        //     intensity.set_color(rgbIntensity);
        // }
};


class Surface{
    public:
        Vector4f ka;
        string name;
        Matrix4f transformation;

        Vector4f center;
        float radius;

        Surface(){
            // do nothing
        }

        Surface(string name, Vector4f ka, Matrix4f transformation){
            // this transformation matrix is from object space to world space;
            this->name = name;
            this->ka= ka;
            this->transformation = transformation;
        }
};

class Sphere : public Surface{
    public:
        // float radius;
        // Vector4f center;
        Sphere() : Surface(){
        }
        Sphere (Vector4f ka_value, Matrix4f transformation_value, Vector4f center, float radius) : Surface ("Sphere", ka_value, transformation_value){
            this->radius = radius;
            this->center = center;
        }
};


//***********************************************
//  Gloabla variables
//***********************************************
static float const ImageLength = 800;
static float const ImageHeight = 600;
static float const fov = 45.0f;
static int const TraceDepth = 0;


int hit_index = -1;



// used for test and debuging
Matrix4f test_translate; 
Vector4f test_ka;
Vector4f test_center;
float test_radius;

Sphere test_sphere;


Vector4f skycolor = Vector4f(0.2, 0.2, 0.2, 0.0);


// list of all surfaces object and light sources etc.
int obj_counter;
// Surface *objects = new Surface[obj_counter];
std::vector<Surface> objects;

int pt_light_counter;
Vector4f *pt_xyz = new Vector4f[pt_light_counter];   // point (x, y, z, 0)
Vector4f *pt_rgb = new Vector4f[pt_light_counter];

int dl_light_counter;
Vector4f *dl_xyz = new Vector4f[dl_light_counter];   // direction (x, y, z, 1)
Vector4f *dl_rgb = new Vector4f[dl_light_counter];

// Trace Depth

// temporary test stats
Vector4f camara_position = Vector4f(0, 0, 4, 1);
Vector4f camara_looking_direction = Vector4f(0, 0, -1, 0);
Vector4f camara_up_direction = Vector4f(0, 1, 0, 0);


// temporary test stats
Vector4f ka = Vector4f(0.2, 0.2, 0.2, 0.0);
Vector4f kd = Vector4f(0.0, 0.0, 0.0, 0.0);
Vector4f ks = Vector4f(0.0, 0.0, 0.0, 0.0);
float p = 20;


//***********************************************
//  Utility functions
//***********************************************

Matrix4f translation(Vector4f a, Vector4f b){
    Vector4f c = b-a;
    Matrix4f translate; translate<<1,0,0,c[0],0,1,0,c[1],0,0,1,c[2],0,0,0,1;
    return translate;
};

Matrix4f scale(Vector4f scaler){
    Matrix4f scales; scales<<scaler[0],0,0,0,0,scaler[1],0,0,0,0,scaler[2],0,0,0,0,1;
    return scales;
};

Matrix4f rotate(Vector4f rotate){
    float x = rotate[0];
    float y = rotate[1];
    float z = rotate[2];
    float angle = rotate[3];
    Matrix4f I; I<<1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;
    if (x==1.0){
        Matrix4f mx; mx<<1,0,0,0,0,cos(angle),(-sin(angle)),0,0, sin(angle), cos(angle),0,0,0,0,1;
        I = mx*I;
    }
    if (y==1.0){
        Matrix4f my; my<<cos(angle),0,sin(angle),0,0,1,0,0,(-sin(angle)),0,cos(angle),0,0,0,0,1;
        I = my*I;
    }
    if (z==1.0){
        Matrix4f mz; mz<<cos(angle),(-sin(angle)),0,0,sin(angle),cos(angle),0,0,0,0,1,0,0,0,0,1;
        I = mz*I;
    }
    return I;
};


Vector4f times(Vector4f a, Vector4f b){
    Vector4f c;
    c[0]=a[0]*b[0];
    c[1]=a[1]*b[1];
    c[2]=a[2]*b[2];
    c[3]=a[3]*b[3];
    return c;
};

Vector4f normalize(Vector4f v) {
    if (v[0]!=0 || v[1]!=0 || v[2]!=0) {
        float sc = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        v[0] = (v[0])/sc;
        v[1] = (v[1])/sc;
        v[2] = (v[2])/sc;
    }
    return v;
};

//coordinate conversion
Matrix4f coordinate_convertor(Matrix4f A, Matrix4f B){
    return B*(A.transpose());
};

//multiplication
Vector4f M_V_Convert(Matrix4f a, Vector4f b){
    return a*b;
};


// Shading
Vector4f diffuseTerm(Vector4f kd, Vector4f intens, Vector4f normal, Vector4f light){
    float dotProduct = light.adjoint()*normal;
    dotProduct = max(dotProduct,(float)0);
    if (dotProduct == (float)0) {
        return Vector4f(0,0,0,0);
    } 
    else{
        return times(kd,intens).adjoint()*dotProduct;
    }
};

Vector4f specularTerm(Vector4f ks, Vector4f intens, Vector4f normal, Vector4f light, Vector4f viewer, float p){
    
    Vector4f refl = light*(float)(-1) + normal*((float)2*(light.adjoint()*normal));
    float dotProduct = refl.adjoint()*viewer;
    dotProduct = max(dotProduct,(float)0);
    if (dotProduct==(float)0){
        return Vector4f(0,0,0,0);
    }else{
        return times(ks,intens)* pow(dotProduct, p);
    }
};

//***********************************************
//  Modules & Sub-Functions
//***********************************************

MatrixXf Find_nearest(Ray, std::vector<Surface>);


Vector4f get_color(Vector4f viewer, Vector4f normal, Vector4f intersect){
    Vector4f R = objects[hit_index].ka;
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    Ray testshadow;
    MatrixXf is_shadow;
    
    if (pt_light_counter != 0) {
        for (int l = 0; l < pt_light_counter; l++) {
            Vector4f pt_light_xyz = pt_xyz[l];
            Vector4f pt_light_rgb = pt_rgb[l];
            Vector4f light = pt_light_xyz - intersect;
            light.normalize();
            
            testshadow = Ray(intersect, light);
            is_shadow = Find_nearest(testshadow, objects);
            
            if(is_shadow==None){
                Vector4f diffuse = diffuseTerm(kd, pt_light_rgb, normal, light);
                
                Vector4f specular = specularTerm(ks, pt_light_rgb, normal, light, viewer, p);
                
                Vector4f ambient = times(ka,pt_light_rgb);
                
                R = R + (diffuse + specular + ambient);
            }
        }
    }
    
    if (dl_light_counter != 0) {
        for (int l = 0; l < dl_light_counter; l++) {
            Vector4f dl_light_rgb = dl_rgb[l];
            Vector4f light = dl_xyz[l];
            light.normalize();
            
            testshadow = Ray(intersect, light);
            is_shadow = Find_nearest(testshadow, objects);
            
            if(is_shadow==None){
                
                Vector4f diffuse = diffuseTerm(kd, dl_light_rgb, normal, light);
                
                Vector4f specular = specularTerm(ks, dl_light_rgb, normal, light, viewer, p);
                
                Vector4f ambient = times(ka,dl_light_rgb);
                
                R = R + (diffuse + specular + ambient);
            }
            
        }
    }
    return R;
};

// PointIntersection
// in obj space
MatrixXf PointIntersection(Ray ray, Surface surface){
    Vector4f e=ray.origin;
    Vector4f d=ray.direction;

    if (ray.direction[3] != 0 || ray.origin[3] != 1){
        cout << "\n";
        cout << ray.direction << endl;
        cout << ray.origin << endl;
        cout << "\n";
    }

    // if (e[3] != 1 || d[3] != 0){
    //             cout << "\n";
    //             // cout << intersection << endl;
    //             cout << "e is " << e << endl;
    //             cout << "d is " << d << endl;
    //             cout << "\n";
    // }

    Vector4f n,intersection,c;
    float t_1,t_2,t_3,t1,t2,t,discriminant,discriminant1,discriminant2,discriminant3,R;
    bool Flag=false;
    if(surface.name=="Sphere"){

        c = ((Sphere*) &surface) -> center;
        R = ((Sphere*) &surface) -> radius;

        if (R != 1){
            printf("%f", R);
        }

        // if (c[0] != 0 || c[1] != 0 || c[2] != 0 || c[3] != 1){
        //    cout << c << endl;
        // }

        discriminant1 = (d.adjoint()*(e-c))*(d.adjoint()*(e-c));
        discriminant2 = (d.adjoint()*d);
        discriminant3 = ((e-c).adjoint()*(e-c)-R*R);
        discriminant = discriminant1-discriminant2*discriminant3;

        if (discriminant<(float)0){
        }

        else if(discriminant==(float)0){
            t_1 = (d.adjoint()*(e-c)); // B/2
            t_2 = (d.adjoint()*d);     // A
            t = -t_1/t_2;
            if (t < 0){
                Flag = false;
            } else {
                Flag = true;
            }
        }

        else{

            t_1 = d.adjoint()*(e-c);
            t_2 = sqrt(discriminant);
            t_3 = (d.adjoint()*d);
            t1 = (-t_1+t_2)/t_3;
            t2 = (-t_1-t_2)/t_3;

            if (t1 < 0){
                Flag = false;
            } else if (t2 < 0){
                Flag = true;
                t = t1;
            } else {
                Flag = true;
                t = t2;
            }    
        }
    

        if (Flag){
            n = ((e+d*t)-c)/R;
            intersection = e+d*t;
            MatrixXf NAndP(4,2); NAndP<<n[0],intersection[0],n[1],intersection[1],n[2],intersection[2],n[3],intersection[3];
            

            // if (intersection[3] != 1){
            //     cout << "\n";
            //     cout << intersection << endl;
            //     cout << "e is " << e << endl;
            //     cout << "d is " << d << endl;
            //     cout << "\n";
            // }
            

            return NAndP;
        }
    } else if (surface.name == "Triangle"){

    }
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    return None;
};

MatrixXf Find_nearest(Ray ray, std::vector<Surface> surface){

    // if (ray.direction[3] != 0 || ray.origin[3] != 1){
    //     cout << "\n";
    //     cout << ray.direction << endl;
    //     cout << ray.origin << endl;
    //     cout << "\n";
    // }

    float t;
    float compare=100000000;
    bool Flag = false;
    Vector4f finalpoint, finalnormal;
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    
    Vector4f returnP, returnN;

    for (int i=0; i < obj_counter; i++){

        // if (surface[i].transformation.inverse() != surface[i].transformation){
        //     cout << surface[i].transformation.inverse() << endl;
        // }

        Vector4f origin = (surface[i].transformation).inverse() * ray.origin;
        Vector4f direction = (surface[i].transformation).inverse() * ray.direction;
        Ray newRay = Ray(origin,direction);

        MatrixXf intersection = PointIntersection(newRay,surface[i]);

        if (intersection!=None){
            Flag = true;
            Vector4f point(intersection(1),intersection(3),intersection(5),intersection(7));
            Vector4f normal(intersection(0),intersection(2),intersection(4),intersection(6));
            finalpoint = surface[i].transformation * point;
            finalnormal = surface[i].transformation * normal;
            t = ray.shootdistance(finalpoint);
            if (t<compare){
                hit_index = i;
                returnP = finalpoint;
                returnN = finalnormal;
                compare = t;
            }
        }
    }
    if (Flag){
        MatrixXf returnValue(4,2); returnValue<<returnN[0],returnP[0],returnN[1], returnP[1],returnN[2],returnP[2],returnN[3], returnP[3];
        return returnValue;
    }
    return None;
}

Vector4f find_reflection(Ray ray, Vector4f normal){

    // if (ray.direction[3] != 0 || ray.origin[3] != 1){
    //     cout << ray.direction << endl;
    //     cout << ray.origin << endl;
    // }

    Vector4f direction = ray.direction;
    float c = -normal.adjoint()*direction;
    float divider = normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2];
    c = c/divider;
    Vector4f Reflect = direction + (2*normal*c);


    return Reflect;
};

//***********************************************
//  Master Funciton
//***********************************************
Vector4f trace(Ray ray, std::vector<Surface> objects, int TraceDepth){

    // if (ray.direction[3] != 0 || ray.origin[3] != 1){
    //     cout << "\n";
    //     cout << ray.direction << endl;
    //     cout << ray.origin << endl;
    //     cout << "\n";
    // }


    Vector4f R = Vector4f(0.0, 0.0, 0.0, 0.0);
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    if (TraceDepth < 0){
        return R;
    }

    // printf("doing ray tracing, finding intersection. . .");
    // cout << "ray is " << ray.direction << endl;

    // find the neares hit
    MatrixXf result = Find_nearest(ray, objects);

    if (result == None){
        return skycolor;
    }

    // printf("hit something\n\n\n\n\n");

    Vector4f normal = Vector4f(result(0),result(2),result(4),result(6));
    Vector4f intersect = Vector4f(result(1),result(3),result(5),result(7));

    // if (normal[3] != 0 || intersect[3] != 1){
    //             cout << "\n";
    //             cout << "fuckfuck fuck fuck";
    //             cout << intersect << endl;
    //             cout << normal << endl;
    //             cout << "\n";
    // }
    // result = Find_nearest(ray, objects);
    
    // generate another ray
    Vector4f rflct = find_reflection(ray, normal);
    Ray reflection_ray = Ray(intersect, rflct);



    // calculate shading
    // shadow test, according to pt_light and dl_light
    
    R = R + get_color(ray.origin, normal, intersect);

    // if (hit_index == -1){
    //     printf("lol");
    // }

    hit_index = -1;

    // reflection
    // R = R + trace(reflection_ray, objects, TraceDepth-1);

    return R;
    
};



//***********************************************
//  Main Funciton
//***********************************************


int main(int args, char* argv[]){




    FreeImage_Initialise();

    FIBITMAP *bitmap = FreeImage_Allocate(WIDTH, HEIGHT, BPP);
    RGBQUAD color;

    if (!bitmap)
        exit(1);


    float aspect_ratio = WIDTH/HEIGHT;
    float fovV = fov/180.0f*PI;
    float vertical_offset = tan(fov/2);
    float horizontal_offset = vertical_offset*aspect_ratio;


    Vector4f dd = camara_position + camara_looking_direction
    Vector4f tt 


    // float fovW = fov/360.0f*PI;
    // float fovH = fovW * (float)HEIGHT/(float)WIDTH;
    // float tanFovW = tan(fovW);
    // float tanFovH = tan(fovH);
    // float zNear = -1.0f;
    // float zFar = -1000.0f;

    // printf("problem2\n");


    // used for testing
    test_translate << 1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f, 
                    0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f;
    test_ka = Vector4f(1.0, 0.0, 0.0, 0.0);
    test_center = Vector4f(0.0, 0.0, 0.0, 1.0);
    test_radius = 1.0f;

    test_sphere = Sphere(test_ka, test_translate, test_center, test_radius);

    // printf("problem3\n");

    obj_counter = 1;
    objects.push_back(test_sphere);

    pt_light_counter = 0;
    // pt_xyz[0] = Vector4f(1.0f, 1.0f, 1.0f, 1.0f);
    // pt_rgb[0] = Vector4f(1.0f, 1.0f, 1.0f, 0.0f);

    dl_light_counter = 0;

    // printf("problem4\n");

    bool gogo = true;

    if (gogo){

    for (int i=0; i<WIDTH; i++){
        for (int j=0; j<HEIGHT; j++){
        //construct a ray

            if (i == 0 && j == 0){
                printf("Ray Tracer Initialized, System Rendering. . .\n");
            }

            if (i == 0 && j == HEIGHT/2){
                printf("Ray Tracing Running, 12.5 percent completed. . .\n");
            }

            if (i == 0 && j == HEIGHT){
                printf("Ray Tracing Running, 25 percent completed. . .\n");
            }

            if (i == WIDTH/2 && j == 0){
                printf("Ray Tracing Running, 37.5 percent completed. . .\n");
            }

            if (i == WIDTH/2 && j == HEIGHT/2){
                printf("Ray Tracing Running, 50 percent completed. . .\n");
            }

            if (i == WIDTH/2 && j == HEIGHT){
                printf("Ray Tracing Running, 62.5 percent completed. . .\n");
            }

            if (i == WIDTH && j == 0){
                printf("Ray Tracing Running, 75 percent completed. . .\n");
            }

            if (i == WIDTH && j == HEIGHT/2){
                printf("Ray Tracing Running, 86.5 percent completed. . .\n");
            }

            if (i == WIDTH && j == HEIGHT){
                printf("Ray Tracing Running, 100 percent completed. . .\n");
            }


            float x = ((2.0f * (float) j) - (float) WIDTH) / (float)WIDTH * tanFovW;
            float y = ((2.0f * (float) i) - (float) HEIGHT) / (float)HEIGHT * tanFovH;

            Vector4f target = Vector4f(x, y, zNear, 0);
            target.normalize();
            Ray initialRay = Ray(camara_position, target);
            
            // if ((initialRay.direction)[3] != 0 || (initialRay.origin)[3] != 1){
            //     cout << "\n";
            //     cout << "fuckfuck fuck fuck";
            //     cout << initialRay.origin << endl;
            //     cout << initialRay.direction << endl;
            //     cout << "\n";
            // }
            
            // printf("    sub_problem1\n");
            Vector4f result = trace(initialRay, objects, TraceDepth);
            // printf("    sub_problem2\n");
            color.rgbRed = result[2]*255;
            color.rgbGreen = result[1]*255;
            color.rgbBlue = result[0]*255;

            // if (color.rgbRed != 0 || color.rgbGreen != 0 || color.rgbBlue != 0){
            //     printf("weird, isn't it? but at least you got values");
            // }

            FreeImage_SetPixelColor (bitmap, i, j, &color);
        }
    }
    }

    // printf("problem5\n");

    // delete [] objects;
    
    // printf("problem6\n");


    if (FreeImage_Save(FIF_PNG, bitmap, "test.png", 0))
        cout << "Image successfully saved!" << endl;
    
    FreeImage_DeInitialise();
}
























