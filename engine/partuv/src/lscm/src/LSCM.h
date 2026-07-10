#ifndef _LSCM_H_
#define _LSCM_H_

#include <vector>
#include <list>

#include "Mesh.h"
#include "Iterators.h"
#include "FormTrait.h"


namespace MeshLib {

class LSCM {
public:
    LSCM(Mesh *mesh);
    ~LSCM();

    int project(Eigen::MatrixXd & v_UV );
    
protected:
    Mesh *m_mesh;
    std::vector<Vertex*> m_fix_vertices;

    void set_coefficients();
    void set_coefficients_parallel();  
    void set_coefficients_faster(); 
    
};

}

#endif
