// helloWrap.c
#include <python2.7/Python.h>
#include<math.h>
#include<stdlib.h> 
#define feature_num 3

double factorial(double value){
  if(value>0){
    return log(value)+factorial(value-1);
  }else{return 0;}
  
}


static PyObject * 
multinomial(PyObject* self, PyObject* args){
  //unsigned int min,max;
  
  int i,j,n;
  double sum;
  PyObject *get_list,*get_list2;
  
  if (PyArg_ParseTuple(args, "OOd",&get_list,&get_list2,&sum)){ 
    int len=PyList_Size(get_list);
    double phi[len],vec[len];
      //n=PyList_Size(get_list);
      if PyList_Check(get_list) {
      for (i=0; i<len; i++){
    
    phi[i] =PyFloat_AsDouble( PyList_GetItem(get_list, (Py_ssize_t)i));
    vec[i] =PyFloat_AsDouble( PyList_GetItem(get_list2, (Py_ssize_t)i));
      }
    }
  //double prob=sum;
  double prob=factorial(sum);
  //double prob=tgamma(sum);
  //printf("%f \n",prob);
  double left=0,right=0;
  for(i=0;i<len;i++){
    double temp;
    if(vec[i]==0){
      temp=1.0;
    }else{
      temp=tgamma(vec[i]);
    }

    left+=log(temp);
    //printf("%d %f \n",i,left);
    n=0;
    for(j=0;j<vec[i];j++){
      n++;
      right +=log(phi[i]);

    }
  }
  //printf("%f %f\n",left,right);
  prob=prob-left + right;
  
    return Py_BuildValue("d",prob);
  }else{return Py_BuildValue("");}

}

/*
static PyObject * 
multinomial(PyObject* self, PyObject* args){
  double value;
  if (PyArg_ParseTuple(args, "d",&value)){ 
    value=factorial(value);
    return Py_BuildValue("d",value);
  }else{return Py_BuildValue("");}

}*/
 

static PyMethodDef multi_distmethods[] = {
  {"multinomial", multinomial, METH_VARARGS,"return prob.\n"},
  {NULL},
};

void initmulti_dist(void){
  Py_InitModule("multi_dist", multi_distmethods);
}

