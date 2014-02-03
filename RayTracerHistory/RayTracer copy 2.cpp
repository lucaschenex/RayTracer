//
//  RayTracer.cpp
//  
//
//  Built by Ian Chen and Betty Chen at Mar 2013
//
//
//  version Mar5 14:44
//



#include "FreeImage.h"
#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include <vector>
#include <math.h>

#define BPP 24
#define PI 3.1415926535897932384626

using namespace std;
using namespace Eigen;

//***********************************************
//  Utility Function Claim
//***********************************************

Vector4f normalize(Vector4f);


//***********************************************
//  Classes
//***********************************************
class Color{
    public:
        Vector3f intensity;
        float r;
        float g;
        float b;
        Color (){
        }
        Color (float r, float g, float b){  
            this->intensity = Vector3f(r, g, b);
            this->r = r;
            this->g = g;
            this->b = b;
        }

        Color operator+ (Color const &c){
        	return Color(r+c.r, g+c.g, b+c.b);
        } 
        Color operator* (Color const &c){
        	return Color(r*c.r, g*c.g, b*c.b);
        }
        Color operator* (float scaler){
        	return Color(r*scaler, g*scaler, b*scaler);
        }
        // Color operator= (Color &c){
        // 	return Color(r, g, b);
        // }
};

class Ray{
    public:
        Vector4f origin;       
        Vector4f direction;     
        Ray(){
        }

        Ray(Vector4f camara, Vector4f direc){
            this->origin = camara;
            this->direction = direc;
        }

        Vector4f shootpoint(float t){
            return this->origin + this->direction * t;
        }

        float get_t(Vector4f point){
        	Vector4f dir = point -origin;
        	for (int i = 0; i < 3; i ++){
        		if (dir[i] > 0.01 && direction[i] > 0.01){
					float t = dir[i]/direction[i];
					if (t > 0){
						return t;
					} else {
						printf(" Error: you have a negative t value. fuck off. \n");
					}
        		}
        	}
        }
};


class Surface{
    public:
        Color ka;
        string name;
        Matrix4f transformation;

        Vector4f center;
        float radius;

        Surface(){
            // do nothing
        }

        Surface(string name, Color ka, Matrix4f transformation){
            // this transformation matrix is from object space to world space;
            this->name = name;
            this->ka = ka;
            this->transformation = transformation;
        }
};

class Sphere : public Surface{
    public:
        Sphere() : Surface(){
        }
        Sphere (Color ka_value, Matrix4f transformation_value, Vector4f center, float radius) : Surface ("Sphere", ka_value, transformation_value){
            this->radius = radius;
            this->center = center;
        }
};

//***********************************************
//  Gloabla variables
//***********************************************
int TraceDepth;
int WIDTH;
int HEIGHT;
float aspect_ratio;
float fov;
const Color skycolor = Color(0.2, 0.2, 0.2);

int hit_index = -1;


// list of all surfaces object and light sources etc.
int obj_counter;
// Surface *objects = new Surface[obj_counter];
std::vector<Surface> objects;

int pt_light_counter;
Vector4f *pt_xyz = new Vector4f[pt_light_counter];   // point (x, y, z, 0)
Color *pt_rgb = new Color[pt_light_counter];

int dl_light_counter;
Vector4f *dl_xyz = new Vector4f[dl_light_counter];   // direction (x, y, z, 1)
Color *dl_rgb = new Color[dl_light_counter];


// temporary test stats
Vector4f camara_position; 
Vector4f camara_looking_direction;
Vector4f camara_up_direction;
Vector4f camara_right_direction;


// temporary test stats
Color ka;
Color kd;
Color ks;
float p;


//***********************************************
//  Utility functions
//***********************************************

Vector4f normalize(Vector4f v) {
	if (v[3] != 0){
		cout << "the vector you want to normalize is not a direction, v is given by: " << v << endl;
	}

    if (abs(v[0]) > 0.001 || abs(v[1]) > 0.001 || abs(v[2]) > 0.001) {
        float sc = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        v[0] = (v[0])/sc;
        v[1] = (v[1])/sc;
        v[2] = (v[2])/sc;
        v[3] = 0;
    }
    return v;
};

Vector4f cross_product(Vector4f a, Vector4f b){
	Vector3f aa = Vector3f(a[0], a[1], a[2]);
	Vector3f bb = Vector3f(b[0], b[1], b[2]);
	Vector3f product = aa.cross(bb);
	Vector4f result = Vector4f(product[0], product[1], product[2], 0);
	return result;
};



// Shading
Color diffuseTerm(Color kd, Color intens, Vector4f normal, Vector4f light){
    float dotProduct = light.adjoint()*normal;
    dotProduct = max(dotProduct,(float)0);
    if (dotProduct == (float)0) {
        return Color(0.0f, 0.0f, 0.0f);
    } 
    else{
        Color result = (kd*intens)*dotProduct;
        if (result.intensity[0] != 0 || result.intensity[1]!=0 || result.intensity[2] != 0){
            cout << result.intensity << endl;
            cout << "\n";
        } 
        return result;
    }
};

Color specularTerm(Color ks, Color intens, Vector4f normal, Vector4f light, Vector4f viewer, float p){
    
    Vector4f refl = light*(-1) + normal*((float)2*(light.adjoint()*normal));
    float dotProduct = refl.adjoint()*viewer;
    // cout<<dotProduct<<endl;
    dotProduct = max(dotProduct,0.0f);
    if (dotProduct==0.0f){
        return Color(0.0f,0.0f,0.0f);
    }else{
        Color result = (ks*intens)* (pow(dotProduct, p));
        if (dotProduct > 1){
            cout << dotProduct<<endl;
        }
        // if (result.intensity[0] > 1 || result.intensity[1] > 1 || result.intensity[2] > 1){
        //     printf ("why you got larger than 1?");
        // }
        return result;
    }
};



MatrixXf Find_nearest(Ray, std::vector<Surface>);


Color get_color(Vector4f viewer, Vector4f normal, Vector4f intersect){
    Color R = ka;
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    Ray testshadow;
    MatrixXf is_shadow;
    
    if (pt_light_counter != 0) {
        for (int l = 0; l < pt_light_counter; l++) {
            Vector4f pt_light_xyz = pt_xyz[l];
            Color pt_light_rgb = pt_rgb[l];
            Vector4f light = pt_light_xyz - intersect;
            light = normalize(light);
            normal = normalize(normal);
            // viewer = normalize(viewer);
            
            // testshadow = Ray(intersect, light);
            // is_shadow = Find_nearest(testshadow, objects);
            
            // if(is_shadow==None){
                Color diffuse = diffuseTerm(kd, pt_light_rgb, normal, light);
                
                Color specular = specularTerm(ks, pt_light_rgb, normal, light, viewer, p);
                // cout<<specular.intensity<<endl;

                R = R + (diffuse + specular);

            // }
        }
    }

    if (dl_light_counter != 0) {
        for (int l = 0; l < dl_light_counter; l++) {
            Color dl_light_rgb = dl_rgb[l];
            Vector4f light = -1 * dl_xyz[l];
            light = normalize(light);
            normal = normalize(normal);
            
            // testshadow = Ray(intersect, light);
            // is_shadow = Find_nearest(testshadow, objects);
            
            // if(is_shadow==None){
                
                Color diffuse = diffuseTerm(kd, dl_light_rgb, normal, light);
                
                Color specular = specularTerm(ks, dl_light_rgb, normal, light, viewer, p);
                
                R = R + (diffuse+specular);
            // }
            
        }
    }
    return R;
};


// Color get_color(Vector4f viewer, Vector4f normal, Vector4f intersect){
// 	return Color(1.0f, 0.0f, 0.0f);
// };

// PointIntersection
// in obj space
MatrixXf PointIntersection(Ray ray, Surface surface){
    Vector4f e=ray.origin;
    Vector4f d=ray.direction;

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

        // if (R != 1){
            // printf("%f", R);
        // }

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
    // } else if (surface.name == "Triangle"){

    // }
    }
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    return None;
};

MatrixXf Find_nearest(Ray ray, std::vector<Surface> surface){
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
            // Flag = true;
            Vector4f point(intersection(4),intersection(5),intersection(6),intersection(7));
            Vector4f normal(intersection(0),intersection(1),intersection(2),intersection(3));
            finalpoint = surface[i].transformation * point;
            finalnormal = surface[i].transformation * normal;
            // t = ray.shootdistance(finalpoint);
            for(int j=0; j<3;j++){
                if ((finalpoint - origin)[j] != 0){
                    t = (finalpoint - origin)[j]/direction[i];
                }
            }
            if (t<compare){
                Flag = true;
                // hit_index = i;
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
};


Color trace(Ray ray,int TraceDepth){
	Color R = Color(0.0, 0.0, 0.0);
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    if (TraceDepth < 0){
        return R;
    }

    // find the neares hit
    MatrixXf result = Find_nearest(ray, objects);

    if (result == None){
        return skycolor;
    }

    // printf("hit something\n\n\n\n\n");

    Vector4f normal = Vector4f(result(0),result(1),result(2),result(3));
    Vector4f intersect = Vector4f(result(4),result(5),result(6),result(7));

   	// cout << "normal is \n" << normal << endl;
   	// cout << "intersect is \n" << intersect << endl;
   	// cout << "\n\n";

    // if (normal[3] != 0 || intersect[3] != 1){
    //             cout << "\n";
    //             cout << "fuckfuck fuck fuck";
    //             cout << intersect << endl;
    //             cout << normal << endl;
    //             cout << "\n";
    // }
    // result = Find_nearest(ray, objects);
    
    // generate another ray
    // Vector4f rflct = find_reflection(ray, normal);
    // Ray reflection_ray = Ray(intersect, rflct);
    // normal = normalize(normal);

    Vector4f viewer = ray.origin-intersect;
    viewer = normalize(viewer);

    Color pp = get_color(viewer, normal, intersect);
    // cout << pp.intensity << endl;
    R = R + pp;
    // cout << R.intensity << endl;

    hit_index = -1;

    return R;
};












//***********************************************
//  Main Funciton
//***********************************************
int main(int args, char* argv[]){

	// specify camara:
	WIDTH = 200;
	HEIGHT = 200;
	aspect_ratio = WIDTH/HEIGHT;
	fov = 45.0f;
	camara_position = Vector4f(0.0f, 0.0f, 0.3f, 1.0f);
	camara_looking_direction = Vector4f (0.0f, 0.0f, -1.0f, 0.0f);
	camara_up_direction = Vector4f (0.0f, 1.0f, 0.0f, 0.0f);
	camara_right_direction = cross_product(camara_looking_direction, camara_up_direction);
	TraceDepth = 0;

	float fovV = fov/180.0f*PI;
	float vertical_offset = tan(fov/2);
	float horizontal_offset = vertical_offset*aspect_ratio;

	float rr = horizontal_offset;
	float ll = -horizontal_offset;
	float tt = vertical_offset;
	float bb = -vertical_offset;


	// specify light sources
    kd = Color(0.521 ,0.72 ,0.2);
	ks = Color(0.262, 0.623, 0.123);
    p = 30;

	pt_light_counter = 1;
	pt_rgb[0] = Color(1.25, 6.43, 6.423);
	pt_xyz[0] = Vector4f(1.512, 1.123, 1.132, 1.0);

    // dl_light_counter = 1;
    // dl_xyz[0] = Vector4f(-0.5, 0.1234, -0.42, 0.0);
    // dl_rgb[0] = Color(0.235, 0.263, 0.233);    



	// draw a sphere:
	float radius = 1.0f;
	ka = Color(0.0, 0.0, 0.0);
	Matrix4f test_translate;
	test_translate << 1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f, 
                    0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f;
	Vector4f center = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
	Sphere test_sphere(ka, test_translate, center, radius);
	

	// initialize world correspondingly
	obj_counter = 1;
	objects.push_back(test_sphere);


    FreeImage_Initialise();

    FIBITMAP *bitmap = FreeImage_Allocate(WIDTH, HEIGHT, BPP);
    RGBQUAD color;

    if (!bitmap)
        exit(1);

    for (int i=0; i<WIDTH; i++){
        for (int j=0; j<HEIGHT; j++){

        	// Ray Generation according to camara geometry
        	float u = ll + (rr-ll)*(i+0.5)/WIDTH;
        	float v = bb + (tt-bb)*(j+0.5)/HEIGHT;

        	Vector4f direction = camara_looking_direction + u * camara_right_direction + v * camara_up_direction;
        	direction = normalize(direction);

        	Vector4f origin = camara_position;

        	Ray initial_ray = Ray (origin, direction);

        	// cout << "initial ray origin\n" << initial_ray.origin << endl;
       		// cout << "initial ray direction\n" << initial_ray.direction << endl;
       		// cout << "\n\n";

        	Color result = trace(initial_ray, TraceDepth);

        	color.rgbRed = (result.intensity[2]*255 > 255 ? 255 : result.intensity[2]*255);
        	color.rgbGreen = result.intensity[1]*255 > 255 ? 255 : result.intensity[1]*255;
       		color.rgbBlue = result.intensity[0]*255 > 255 ? 255 : result.intensity[0]*255;

			FreeImage_SetPixelColor (bitmap, i, j, &color);
        }
    }

    if (FreeImage_Save(FIF_PNG, bitmap, "test.png", 0))
        cout << "Image successfully saved!" << endl;
    
    FreeImage_DeInitialise();
}


