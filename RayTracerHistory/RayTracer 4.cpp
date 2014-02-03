//
//  RayTracer.cpp
//  
//
//  Built by Ian Chen and Betty Chen at Mar 2013
//
//
//  version Mar6 23:20
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
Vector4f error_avoid(Vector4f, Vector4f);
Vector4f cross_product(Vector4f, Vector4f);


//***********************************************
//  Classes
//***********************************************
class Color{
    public:
        float r;
        float g;
        float b;
        Vector3f intensity;
        Color (){
        }
        Color (float r, float g, float b){
            this->r = r;
            this->g = g;
            this->b = b;
            this->intensity = Vector3f(r, g, b);
        }
    ~ Color(){}

        Color operator+ (Color const &c){
        	return Color(r+c.r, g+c.g, b+c.b);
        } 
        Color operator* (Color const &c){
        	return Color(r*c.r, g*c.g, b*c.b);
        }
        Color operator* (float scaler){
        	return Color(r*scaler, g*scaler, b*scaler);
        }
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
        		if (dir[i] !=0 && direction[i] !=0){
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


class Material {
    public:
        Color ka;
        Color kr;
        Color kd;
        Color ks;
        float sp;
        Color emmision;
        Material(){
            // do nothing
        }
        Material(Color ambient, Color diffuse, Color specular, float shiness, Color reflection = Color(0,0,0), Color emmision = Color(0.0, 0.0, 0.0)){
            this->ka = ambient;
            this->kd = diffuse;
            this->ks = specular;
            this->sp = shiness;
            this->kr = reflection;
            this->emmision = emmision;
        }
};


class Vertex{
    public:
        Vector4f coordinate;
        Vertex(){
        }
        Vertex(Vector4f position) {
            this->coordinate = position;

        }
        Vector4f operator+ (Vertex vplus){
            Vector4f sum = coordinate + vplus.coordinate;
            sum[3] = 0;
            return sum;
        }

        Vector4f operator- (Vertex vminus){
            Vector4f dif = coordinate - vminus.coordinate;
            dif[3] = 0;
            return dif;
        }
};

class VertexNormal : public Vertex{
    public:
        Vector4f normal;
        VertexNormal(){
        }
        VertexNormal(Vector4f coordinate, Vector4f normal) : Vertex(coordinate){
            if (normal[3] != 0){
                cout << "note that the normal has to be a direction to maintain consistency" << endl;
            }
            this->normal = normal;
        }
};


class Surface{
    public:
        string name;
        Matrix4f transformation;
        Material material;

        // for spheres only;
        Vector4f center;
        float radius;

        // for regular triangles only
        Vertex v1;
        Vertex v2;
        Vertex v3;
        Vector4f normal;

        // for triangles with normals only
        VertexNormal vn1;
        VertexNormal vn2;
        VertexNormal vn3;

        Surface(){
            // do nothing
        }

        Surface(string name, Matrix4f transformation, Material m){
            // this transformation matrix is from object space to world space;
            this->name = name;
            this->transformation = transformation;
            this->material = m;
        }
};

class Sphere : public Surface{
    public:
        Sphere() : Surface(){
        }
        Sphere (Matrix4f transformation_value, Vector4f center, float radius, Material m) : Surface ("Sphere", transformation_value, m){
            this->radius = radius;
            this->center = center;
        }
    ~Sphere(){}
};

class triangle : public Surface{
    public:
        triangle(){
        }
        triangle(Vertex v1, Vertex v2, Vertex v3, Matrix4f transformation, Material m) : Surface("Triangle", transformation, m){
            this->v1 = v1;
            this->v2 = v2;
            this->v3 = v3;

            Vector4f a1 = v2 - v1;
            Vector4f a2 = v3 - v1;
            Vector4f normal_direction = cross_product(a1, a2);
            this->normal = normalize(normal_direction);
        }
};


float find_area(Vector4f v1, Vector4f v2, Vector4f v3){
    Vector4f a = v2 - v1;
    Vector4f b = v3 - v1;
    Vector4f cx = cross_product(a, b);
    float area = sqrt(pow(cx[0], 2)+pow(cx[1], 2)+pow(cx[2], 2));
    return area/2;
};

class triangleNormal : public Surface{
    public:
        triangleNormal(){
        }
        triangleNormal(VertexNormal vn1, VertexNormal vn2, VertexNormal vn3, Matrix4f transformation, Material m) : Surface("TriangleNormal", transformation, m){
            this->vn1 = vn1;
            this->vn2 = vn2;
            this->vn3 = vn3;
        }



        Vector4f getNormal(Vector4f point){
            float AreaT = find_area(vn1.coordinate, vn2.coordinate, vn3.coordinate);
            float AreaB = find_area(vn1.coordinate, point, vn3.coordinate);
            float AreaC = find_area(vn1.coordinate, point, vn2.coordinate);
            float AreaA = find_area(vn2.coordinate, point, vn3.coordinate);

            float c1 = AreaA/AreaT;
            float c2 = AreaB/AreaT;
            float c3 = AreaC/AreaT;

            float nx = vn1.normal[0] * c1 + vn2.normal[0] * c2 + vn3.normal[0] * c3;
            float ny = vn1.normal[1] * c1 + vn2.normal[1] * c2 + vn3.normal[1] * c3;
            float nz = vn1.normal[2] * c1 + vn2.normal[2] * c2 + vn3.normal[2] * c3;

            Vector4f result = Vector4f(nx, ny, nz, 0);
            return result;
            
        }
};




//     ~triangle();

//     /* data */
// };

//***********************************************
//  Gloable variables
//***********************************************
int TraceDepth;
int WIDTH;
int HEIGHT;
float aspect_ratio;
float fov;
const Color skycolor = Color(0.0, 0.0, 0.0);

int hit_index = -1;


// list of all surfaces object and light sources etc.
int obj_counter;
// Surface *objects = new Surface[obj_counter];
std::vector<Surface> objects;

int pt_light_counter;
int dl_light_counter;
std::vector<Color> pt_rgb;
std::vector<Vector4f> pt_xyz;

std::vector<Color> dl_rgb;
std::vector<Vector4f> dl_xyz;
//Vector4f *pt_xyz = new Vector4f[pt_light_counter];   // point (x, y, z, 1)
//Color *pt_rgb = new Color[pt_light_counter];
//

//Vector4f *dl_xyz = new Vector4f[dl_light_counter];   // direction (x, y, z, 0)
//Color *dl_rgb = new Color[dl_light_counter];


// temporary test stats
Vector4f camara_position; 
Vector4f camara_looking_direction;
Vector4f camara_up_direction;
Vector4f camara_right_direction;


// temporary test stats
Color Ambient_input;
Color Diffuse_input;
Color Specular_input;
float Shiness_input;
Color reflection_input;


//***********************************************
//  Utility functions
//***********************************************

Vector4f error_avoid(Vector4f point, Vector4f normal){
    return point + normal*0.01;
};

Vector4f normalize(Vector4f v) {
	if (v[3] != 0){
		cout << "the vector you want to normalize is not a direction, v is given by: " << v << endl;
	}

    if (v[0] !=0|| v[1] !=0 || v[2] !=0) {
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
        return (kd*intens)*dotProduct;
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
    Color R = Color(0,0,0);
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    Ray testshadow;
    MatrixXf is_shadow;
    Color ka = (objects[hit_index].material).ka;
    Color kd = (objects[hit_index].material).kd;
    Color ks = (objects[hit_index].material).ks;
    float p = (objects[hit_index].material).sp;

    
    if (pt_light_counter != 0) {
        for (int l = 0; l < pt_light_counter; l++) {
            Vector4f pt_light_xyz = pt_xyz[l];
            Color pt_light_rgb = pt_rgb[l];
            Vector4f light = pt_light_xyz - intersect;
            light = normalize(light);
            normal = normalize(normal);
            
            
            testshadow = Ray(error_avoid(intersect,normal), light);
            is_shadow = Find_nearest(testshadow, objects);
            
            if(is_shadow==None){
                Color diffuse1 = diffuseTerm(kd, pt_light_rgb, normal, light);
                
                Color specular1 = specularTerm(ks, pt_light_rgb, normal, light, viewer, p);
                // cout<<specular.intensity<<endl;

                R = R + (diffuse1 + specular1);

            }
        }
    }

    if (dl_light_counter != 0) {
    
        for (int l = 0; l < dl_light_counter; l++) {
            Color dl_light_rgb = dl_rgb[l];
//            cout<<l<<endl;
            
//            cout<<dl_light_rgb.b<<endl;
//            cout << dl_light_rgb.intensity << endl;
//            cout << "\n" << endl;
            
            Vector4f light1 = -dl_xyz[l];
            
            
            light1 = normalize(light1);
//
            
            normal = normalize(normal);
            
//            cout<<light1<<endl;
             testshadow = Ray(error_avoid(intersect,normal), light1);
             is_shadow = Find_nearest(testshadow, objects);
            
             if(is_shadow==None){
//            cout<<dl_light_rgb.intensity<<endl;
                Color diffuse = diffuseTerm(kd, dl_light_rgb, normal, light1);
                
                Color specular = specularTerm(ks, dl_light_rgb, normal, light1, viewer, p);
            
//                cout<<diffuse.intensity<<endl;
                R = R + (diffuse + specular);
                 
             }
//                cout << "plused 1" << endl;
            
        }
    }
    return R +ka;
};



// PointIntersection
// in obj space
MatrixXf PointIntersection(Ray ray, Surface surface){
    Vector4f e=ray.origin;
    Vector4f d=ray.direction;

    Vector4f n,intersection,c;
    float t_1,t_2,t_3,t1,t2,t,discriminant,discriminant1,discriminant2,discriminant3,R;
    bool Flag=false;
    if(surface.name=="Sphere"){

        c = ((Sphere*) &surface) -> center;
        R = ((Sphere*) &surface) -> radius;


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
    }
    
    if (Flag){
        n = ((e+d*t)-c)/R;
        intersection = e+d*t;
        MatrixXf NAndP(4,2); NAndP<<n[0],intersection[0],n[1],intersection[1],n[2],intersection[2],n[3],intersection[3];
        return NAndP;
        }
    
    else if (surface.name=="Triangle" || surface.name=="TriangleNormal"){
        Vector4f v0 = surface.v1.coordinate;
        Vector4f v1 = surface.v2.coordinate;
        Vector4f v2 = surface.v3.coordinate;
            
        Vector4f N = cross_product((v1-v0),(v2-v0));
        float is_parallel = N.adjoint()*d;
        if (is_parallel!=0.0){
            float D = -N.adjoint()*v0;
            float tt = -(N.adjoint()*e + D) / (N.adjoint()*d);
            if (tt>0){
                Vector4f P = e+tt*d;
                Vector4f C;
                Vector4f edge0 = v1-v0;
                Vector4f pv0 = P-v0;
                C = cross_product(edge0,pv0);
                if (N.adjoint()*C>0 or N.adjoint()*C==0){
                    Vector4f edge1 = v2-v1;
                    Vector4f pv1 = P-v1;
                    C = cross_product(edge1,pv1);
                    if (N.adjoint()*C>0 or N.adjoint()*C==0){
                        Vector4f edge2 = v0-v2;
                        Vector4f pv2 = P-v2;
                        C = cross_product(edge2,pv2);
                        if (N.adjoint()*C>0 or N.adjoint()*C==0){
                            MatrixXf returnV(4,2); returnV<<N[0],P[0],N[1],P[1],N[2],P[2],0,1;
                            return returnV;
                        }
                    }
                }
            }   
        }
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

        Vector4f origin = (surface[i].transformation).inverse()* ray.origin;
        
        Vector4f direction = (surface[i].transformation).inverse()* ray.direction;
        
        direction = normalize(direction);
        
        
        
        
        Ray newRay = Ray(origin,direction);

        MatrixXf intersection = PointIntersection(newRay,surface[i]);

        if (intersection!=None){
            // Flag = true;
            Vector4f point(intersection(4),intersection(5),intersection(6),intersection(7));
            Vector4f normal(intersection(0),intersection(1),intersection(2),intersection(3));
            finalpoint = surface[i].transformation* point;
            finalnormal = surface[i].transformation* normal;
            finalnormal = normalize(finalnormal);
            t = ray.get_t(finalpoint);
            if (t>0 && t<compare){
                Flag = true;
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
};

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

Color trace(Ray ray,int TraceDepth){
	Color R = Color(0.0, 0.0, 0.0);
    MatrixXf None(4,2); None<<0,0,0,0,0,0,0,0;
    if (TraceDepth < 0){
        return R;
    }

    // find the neares hit
    MatrixXf result = Find_nearest(ray, objects);

    Color kr = objects[hit_index].material.kr;
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
    normal = normalize(normal);
    Vector4f rflct = find_reflection(ray, normal);
    rflct = normalize(rflct);
    
    Ray reflection_ray = Ray(error_avoid(intersect,normal), rflct);
    

    Vector4f viewer = ray.origin-intersect;
    viewer = normalize(viewer);

    Color pp = get_color(viewer, normal, intersect);
    // cout << pp.intensity << endl;
    R = R + pp;
    // cout << R.intensity << endl;
    
    hit_index = -1;
    R = R + kr*trace(reflection_ray,TraceDepth-1);
    return R;
};


//***********************************************
//  Main Funciton
//***********************************************
int main(int args, char* argv[]){

	// specify camara:
//	WIDTH = 200;
//	HEIGHT = 200;
//	aspect_ratio = WIDTH/HEIGHT;
//	fov = 36.8f;
//	camara_position = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
//	camara_looking_direction = Vector4f (0.0f, 0.0f, -1.0f, 0.0f);
//	camara_up_direction = Vector4f (0.0f, 1.0f, 0.0f, 0.0f);
//	camara_right_direction = cross_product(camara_looking_direction, camara_up_direction);
//	TraceDepth = 5;
//
//	float fovV = fov/180.0f*PI;
//	float vertical_offset = tan(fovV/2);
//    
//	float horizontal_offset = vertical_offset*aspect_ratio;
//
//	float rr = horizontal_offset;
//	float ll = -horizontal_offset;
//	float tt = vertical_offset;
//	float bb = -vertical_offset;
//    
//
//
//	// specify light sources
////	pt_light_counter = 1;
////	pt_rgb[0] = Color(0.125, 0.643, 0.6423);
////	pt_xyz[0] = Vector4f(1.512, 1.123, 1.132,1.0);
//
////  dl_light_counter = 1;
////  dl_xyz[0] = Vector4f(-0.5, 0.1234, -0.42, 0.0);
////  dl_rgb[0] = Color(0.235, 0.263, 0.233);
//
////    dl_light_counter = 1;
////    dl_xyz[0] = Vector4f(3, 3, 3, 0.0);
////    dl_rgb[0] = Color(0.5,0.9, 0.5);
//
////
//    dl_light_counter = 2;
////    dl_light_counter = 1;
////	pt_rgb.push_back(Color(1,1,1));
////	pt_xyz.push_back(Vector4f(1,1,1,1.0));
//
//    dl_light_counter = 2;
//    //    dl_light_counter = 1;
//    //	pt_rgb.push_back(Color(1,1,1));
//    //	pt_xyz.push_back(Vector4f(1,1,1,1.0));
//    
//    dl_rgb.push_back(Color(1,1.0f,1.0f));
//    dl_xyz.push_back(Vector4f(0.5773, -0.5773, -0.5773,0));
//    
//    dl_xyz.push_back(Vector4f(-0.57735027, 0.57735027, 0.57735027,0));
//    dl_rgb.push_back(Color(1,1, 1));
//    
//    //    pt_light_counter = 2;
//    //    pt_xyz.push_back(Vector4f(4, 3, -18,1));
//    //    pt_rgb.push_back(Color(1,1, 1));
//    //
//    //    pt_xyz[1] = Vector4f(-4, 8, -15, 1);
//    //    pt_rgb[1] = Color(1,1, 1);
//    
//    
//    
//    
//    
//    
//	// draw a sphere:
//	float radius = 1.0f;
//	Ambient_input = Color(0.1, 0.1, 0.1);
//    Diffuse_input = Color(1, 0, 0);
//    Specular_input = Color(1, 1, 1);
//    Shiness_input = 50;
//    reflection_input = Color(0.9,0.9,0.9);
//    
//	Matrix4f test_translate;
//	test_translate << 1.0f, 0.0f, 0.0f, 0.0f,
//    0.0f, 2.0f, 0.0f, 0.0f,
//    0.0f, 0.0f, 1.0f, 0.0f,
//    0.0f, 0.0f, 0.0f, 1.0f;
//	Vector4f center = Vector4f(0.0f, 0.0f, -17.0f, 1.0f);
//    Material mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
//    
//    Sphere test_sphere = Sphere(test_translate, center, radius*2, mtrl);
//    
//    
//    Vector4f center2 = Vector4f(0,4,-17,1);
//    reflection_input = Color(0.9,0.9,0.9);
//    
//    Diffuse_input = Color(0, 1, 0);
//    Specular_input = Color (1, 1, 1);
//    Shiness_input = 50;
//    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
//    Sphere test_sphere2(test_translate, center2, radius*1.5, mtrl);
//    
//    
//    Vector4f center3 = Vector4f(0,-4,-17,1);
//    reflection_input = Color(0.9,0.9,0.9);
//    
//    Diffuse_input = Color(0, 0, 1);
//    Specular_input = Color (1, 1, 1);
//    Shiness_input = 50;
//    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
//    Sphere test_sphere3(test_translate, center3, radius*1.5, mtrl);
//    
//    Vector4f center4 = Vector4f(4,0,-17,1);
//    reflection_input = Color(0.9,0.9,0.9);
//    
//    Diffuse_input = Color(1, 1, 0);
//    Specular_input = Color (1, 1, 1);
//    Shiness_input = 50;
//    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
//    Sphere test_sphere4(test_translate, center4, radius*1.5, mtrl);
//    
//    
//    Vector4f center5 = Vector4f(-4,0,-17,1);
//    reflection_input = Color(0.9,0.9,0.9);
//    
//    Diffuse_input = Color(0, 1, 1);
//    Specular_input = Color (1, 1, 1);
//    Shiness_input = 50;
//    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
//    Sphere test_sphere5(test_translate, center5, radius*1.5, mtrl);
//    
//    
//	// initialize world correspondingly
//	obj_counter = 5;
//	objects.push_back(test_sphere);
//    objects.push_back(test_sphere2);
//    objects.push_back(test_sphere3);
//    objects.push_back(test_sphere4);
//    objects.push_back(test_sphere5);

    // specify camara:
    WIDTH = 80;
    HEIGHT = 80;
    aspect_ratio = WIDTH/HEIGHT;
    fov = 36.8f;
    camara_position = Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
    camara_looking_direction = Vector4f (0.0f, 0.0f, -1.0f, 0.0f);
    camara_up_direction = Vector4f (0.0f, 1.0f, 0.0f, 0.0f);
    camara_right_direction = cross_product(camara_looking_direction, camara_up_direction);
    TraceDepth = 5;
    
    float fovV = fov/180.0f*PI;
    float vertical_offset = tan(fovV/2);
    
    float horizontal_offset = vertical_offset*aspect_ratio;
    
    float rr = horizontal_offset;
    float ll = -horizontal_offset;
    float tt = vertical_offset;
    float bb = -vertical_offset;
    
    
    
    // specify light sources
    //	pt_light_counter = 1;
    //	pt_rgb[0] = Color(0.125, 0.643, 0.6423);
    //	pt_xyz[0] = Vector4f(1.512, 1.123, 1.132,1.0);
    
    //  dl_light_counter = 1;
    //  dl_xyz[0] = Vector4f(-0.5, 0.1234, -0.42, 0.0);
    //  dl_rgb[0] = Color(0.235, 0.263, 0.233);
    
    //    dl_light_counter = 1;
    //    dl_xyz[0] = Vector4f(3, 3, 3, 0.0);
    //    dl_rgb[0] = Color(0.5,0.9, 0.5);
    
    //
    // specify light sources
    //	pt_light_counter = 1;
    //	pt_rgb[0] = Color(0.125, 0.643, 0.6423);
    //	pt_xyz[0] = Vector4f(1.512, 1.123, 1.132,1.0);
    
    //  dl_light_counter = 1;
    //  dl_xyz[0] = Vector4f(-0.5, 0.1234, -0.42, 0.0);
    //  dl_rgb[0] = Color(0.235, 0.263, 0.233);
    
    //    dl_light_counter = 1;
    //    dl_xyz[0] = Vector4f(3, 3, 3, 0.0);
    //    dl_rgb[0] = Color(0.5,0.9, 0.5);
    
    //
    dl_light_counter = 2;
    //    dl_light_counter = 1;
    //	pt_rgb.push_back(Color(1,1,1));
    //	pt_xyz.push_back(Vector4f(1,1,1,1.0));
    
    dl_rgb.push_back(Color(1,1.0f,1.0f));
    dl_xyz.push_back(Vector4f(0.5773, -0.5773, -0.5773,0));
    
    dl_xyz.push_back(Vector4f(0.57735027, 0.57735027, -0.57735027,0));
    dl_rgb.push_back(Color(0,0, 1));
    
    //    pt_light_counter = 2;
    //    pt_xyz.push_back(Vector4f(4, 3, -18,1));
    //    pt_rgb.push_back(Color(1,1, 1));
    //
    //    pt_xyz[1] = Vector4f(-4, 8, -15, 1);
    //    pt_rgb[1] = Color(1,1, 1);
    
    
    
    
    
    
    // draw a sphere:
    float radius = 1.0f;
    Ambient_input = Color(0.1, 0.1, 0.1);
    Diffuse_input = Color(1, 0, 1);
    Specular_input = Color(1, 1, 1);
    Shiness_input = 50;
    reflection_input = Color(0.0,0.0,0.0);
    
    Matrix4f test_translate;
    test_translate << 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f;
    Vector4f center = Vector4f(0.0f, 0.0f, -20.0f, 1.0f);
    Material mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
    
    Sphere test_sphere = Sphere(test_translate, center, radius*3, mtrl);
    
    
    Vector4f center2 = Vector4f(-2,2,-15,1);
    reflection_input = Color(0.0,0.0,0.0);
    
    Diffuse_input = Color(1, 1, 0);
    Specular_input = Color (1, 1, 1);
    Shiness_input = 50;
    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
    Sphere test_sphere2(test_translate, center2, radius, mtrl);
    
    
    Vector4f center3 = Vector4f(0,-4,-17,1);
    reflection_input = Color(0.9,0.9,0.9);
    
    Diffuse_input = Color(0, 0, 1);
    Specular_input = Color (1, 1, 1);
    Shiness_input = 50;
    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
    Sphere test_sphere3(test_translate, center3, radius*1.5, mtrl);
    
    Vector4f center4 = Vector4f(-2,-2,-15,1);
    reflection_input = Color(0.0,0.0,0.0);
    
    Diffuse_input = Color(0, 1, 1);
    Specular_input = Color (1, 1, 1);
    Shiness_input = 50;
    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
    Sphere test_sphere4(test_translate, center4, radius, mtrl);
    
    
    Vector4f center5 = Vector4f(-4,0,-17,1);
    reflection_input = Color(0.9,0.9,0.9);
    
    Diffuse_input = Color(0, 1, 1);
    Specular_input = Color (1, 1, 1);
    Shiness_input = 50;
    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
    Sphere test_sphere5(test_translate, center5, radius*1.5, mtrl);
    
    
    
    Vertex v0(Vector4f(5,5,-17,1));
    Vertex v1(Vector4f(1,4,-20,1));
    Vertex v2(Vector4f(6,-1,-20,1));
    
    reflection_input = Color(1,1,1);
    
    Diffuse_input = Color(0.1, 0.1, 0.1);
    Specular_input = Color (1, 1, 1);
    Shiness_input = 50;
    mtrl = Material(Ambient_input, Diffuse_input, Specular_input, Shiness_input, reflection_input);
    triangle test_triangle1(v0,v1,v2, test_translate, mtrl);
    
    
    // initialize world correspondingly
    obj_counter = 4;
    objects.push_back(test_sphere);
    objects.push_back(test_sphere2);
    objects.push_back(test_sphere4);
    //    objects.push_back(test_sphere4);
    //    objects.push_back(test_sphere5);
    objects.push_back(test_triangle1);
    
    
//    cout << dl_rgb[0].intensity << endl;


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






//future issue: 
        //2.transformation of primitives
        //4.parser
        //5.directional light direction?????
       




