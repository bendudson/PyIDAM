/******************************************************************
 * IDAM module for Python
 *
 * Access to IDA and MDS+ data, using D.G.Muir's IDAM library
 *
 * Known issues:
 * - Hangs if server cannot be contacted
 * - This code is not thread-safe, despite being a shared library
 *
 * Released July 2009 under the BSD license:
 *
 * Copyright (c) 2009, B.Dudson, University of York
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. Neither the name of the University of York nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************/

/* NB: Python.h must be included first */
#include <Python.h>

/* For defining types */
#include "structmember.h"

/* For numeric arrays */
/*#include "Numeric/arrayobject.h" */
#include "numpy/arrayobject.h"

/* IDAM library */
#include "idamclientserver.h"
#include "idamclient.h"

static PyObject*
idam_test(PyObject *self, PyObject *args)
{
  const char *str;
  int ret;
  if(!PyArg_ParseTuple(args, "s", &str))
    return NULL;

  printf("String: %s\n", str);

  ret = 1;
  return Py_BuildValue("i", ret);
}

static PyObject*
idam_setHost(PyObject *self, PyObject *args)
{
  const char* host;
  int port = -1;

  if(!PyArg_ParseTuple(args, "s|i", &host, &port))
    // Hostname, optional port
    return NULL;
  
  if(port > 0)
    putIdamServerPort(port);
  
  putIdamServerHost(host);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
idam_setPort(PyObject *self, PyObject *args)
{
  int port;
  if(!PyArg_ParseTuple(args, "i", &port))
    return NULL;
  
  putIdamServerPort(port);

  Py_INCREF(Py_None);
  return Py_None;
}

/************************************************************
 * Simple true/false properties for Client/Server behavior
 ************************************************************/

static PyObject*
idam_setProperty(PyObject *self, PyObject *args)
{
  const char* prop;
  int val;
  
  val = 1; /* By default, set to true */

  if(!PyArg_ParseTuple(args, "s|i", &prop, &val)) {
    return NULL;
  }
  
  if(val) {
    setIdamProperty(prop);
  }else
    resetIdamProperty(prop);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
idam_getProperty(PyObject *self, PyObject *args)
{
  const char* prop;
  int val;

  if(!PyArg_ParseTuple(args, "s", &prop))
    return NULL;

  val = getIdamProperty(prop);

  return Py_BuildValue("i", val);
}

/************************************************************
 * Low-level routines
 ************************************************************/

static PyObject*
idam_getAPI(PyObject *self, PyObject *args)
{
  int handle;
  const char *data, *source;
  
  if(!PyArg_ParseTuple(args, "ss", &data, &source))
    return NULL;
  
  handle = idamGetAPI(data, source);

  if(!getIdamSourceStatus(handle)) {
    fprintf(stderr, "IDAM error: %s\n", getIdamErrorMsg(handle));
  }
  
  return Py_BuildValue("i", handle);
}

static PyObject*
idam_freeAPI(PyObject *self, PyObject *args)
{
  int handle;
  
  if(!PyArg_ParseTuple(args, "i", &handle))
    return NULL;

  idamFree(handle);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
idam_readData(PyObject *self, PyObject *args)
{
  int handle;
  int data_n;
  int rank, order;
  npy_intp dimsize[8];
  int i;
  PyArrayObject *result;

  if(!PyArg_ParseTuple(args, "i", &handle))
    return NULL;

  // Get the size of the data array

  if((data_n = getIdamDataNum(handle)) <= 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  
  // Get the rank and order of the data
  rank = getIdamRank(handle);
  order = getIdamOrder(handle);
  
  //printf("Size = %d\nRank = %d\nOrder = %d\n", data_n, rank, order);

  for(i=0;i<rank;i++) {
    dimsize[i] = getIdamDimNum(handle, i); 
    //printf("Dim %d: %d\n", i, dimsize[i]);
  }
  
  //result = (PyArrayObject*) PyArray_FromDims(rank,dimsize,PyArray_FLOAT); // Depreciated
  result = (PyArrayObject*) PyArray_SimpleNew(rank,dimsize,PyArray_FLOAT);
  if (result == NULL) {
    return NULL;
  }
  
  getIdamFloatData(handle, (float *)(result->data));
  
  return PyArray_Return(result);
}

/************************************************************
 * IDAM dimension members
 ************************************************************/

/* Define data type */
typedef struct {
  PyObject_HEAD
  
  PyObject *label; /* Short label */
  PyObject *units; 
  PyObject *data;  /* NumPy array */

  PyObject *errl;  /* NumPy array of low-side errors */
  PyObject *errh;  /* NumPy array of high-side errors */

} idam_Dimension;

/* Members of the type */
static PyMemberDef idam_DimensionMembers[] = {
  {"label", T_OBJECT_EX, offsetof(idam_Dimension, label), 0,
   "Short label"},
  {"units", T_OBJECT_EX, offsetof(idam_Dimension, units), 0,
   "units"},
  {"data", T_OBJECT_EX, offsetof(idam_Dimension, data), 0,
   "NumPy array of dimension values"},
  {"errl", T_OBJECT_EX, offsetof(idam_Dimension, errl), 0,
   "NumPy array of low-side errors"},
  {"errh", T_OBJECT_EX, offsetof(idam_Dimension, errh), 0,
   "NumPy array of high-side errors"},
  {NULL}  /* Sentinel */
};

/************************************************************
 * IDAM dimension methods
 ************************************************************/

/* Free memory */
static void
Dimension_dealloc(idam_Dimension* self)
{
  Py_XDECREF(self->label);
  Py_XDECREF(self->units);
  
  Py_XDECREF(self->data);
  
  Py_XDECREF(self->errl);
  Py_XDECREF(self->errh);

  self->ob_type->tp_free((PyObject*)self);
}

/* Create a new instance (NOT initialisation) */
static PyObject *
Dimension_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  idam_Dimension *self;

  self = (idam_Dimension *)type->tp_alloc(type, 0);
  if (self != NULL) {
    
    self->label = PyString_FromString("No data");
    if (self->label == NULL) {
      Py_DECREF(self);
      return NULL;
    }

    self->units = PyString_FromString("No units");
    if (self->units == NULL) {
      Py_DECREF(self);
      return NULL;
    }

    Py_INCREF(Py_None);
    self->data = Py_None;
    
    Py_INCREF(Py_None);
    self->errl = Py_None;

    Py_INCREF(Py_None);
    self->errh = Py_None;
  }
  
  return (PyObject *)self;
}

/* Methods */
static PyMethodDef idam_DimensionMethods[] = {
    {NULL}  /* Sentinel */
};

static PyTypeObject idam_DimensionType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "idam.Dimension",          /*tp_name*/
    sizeof(idam_Dimension),    /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Dimension_dealloc,  /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "IDAM dimension objects",  /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    idam_DimensionMethods,     /* tp_methods */
    idam_DimensionMembers,     /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    Dimension_new,             /* tp_new */
};

/************************************************************
 * IDAM data members
 ************************************************************/

/* Define data type */
typedef struct {
  PyObject_HEAD
  PyObject *name;   /* Name of the data (e.g. 'amc_plasma current' */
  PyObject *source; /* Source of the data (e.g. '15100') */
  
  PyObject *label;  /* Data label */
  PyObject *units;  /* Units */
  PyObject *desc;   /* Description */

  PyObject *dim;    /* List of dimensions */
  int order;        /* Which dimension is time */
  
  PyObject *time;   /* Refers to same NumPy array as dim[order]->data */

  PyObject *errl; /* Error on the low side */
  PyObject *errh;  /* Error on the high side */
  
  PyObject *data;
} idam_Data;

/* Members of the type */
static PyMemberDef idam_DataMembers[] = {
  {"name", T_OBJECT_EX, offsetof(idam_Data, name), 0,
   "Name used to request the data"},
  {"source", T_OBJECT_EX, offsetof(idam_Data, source), 0,
   "Source of the data"},

  {"label", T_OBJECT_EX, offsetof(idam_Data, label), 0,
   "Data label"},
  {"units", T_OBJECT_EX, offsetof(idam_Data, units), 0,
   "Data units"},
  {"desc", T_OBJECT_EX, offsetof(idam_Data, desc), 0,
   "Longer description of the data"},

  {"dim", T_OBJECT_EX, offsetof(idam_Data, dim), 0,
   "Dimensions"},

  {"order", T_INT, offsetof(idam_Data, order), 0,
   "Index of time dimension"},

  {"time", T_OBJECT_EX, offsetof(idam_Data, time), 0,
   "Time values. Same as dim[order].data"},

  {"errl", T_OBJECT_EX, offsetof(idam_Data, errl), 0,
   "Error on the low side"},
  {"errh", T_OBJECT_EX, offsetof(idam_Data, errh), 0,
   "Error on the high side"},

  {"data", T_OBJECT_EX, offsetof(idam_Data, data), 0,
   "NumPy data array"},
  {NULL}  /* Sentinel */
};

/************************************************************
 * IDAM data methods
 ************************************************************/

/* Free memory */
static void
Data_dealloc(idam_Data* self)
{
  Py_XDECREF(self->name);
  Py_XDECREF(self->source);

  Py_XDECREF(self->label);
  Py_XDECREF(self->units);
  Py_XDECREF(self->desc);

  Py_XDECREF(self->dim);

  Py_XDECREF(self->time);

  Py_XDECREF(self->errl);
  Py_XDECREF(self->errh);

  Py_XDECREF(self->data);
  
  self->ob_type->tp_free((PyObject*)self);
}

/* Create a new instance (NOT initialisation) */
static PyObject *
Data_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  idam_Data *self;

  self = (idam_Data *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->name = PyString_FromString("");
    if (self->name == NULL) {
      Py_DECREF(self);
      return NULL;
    }
    
    self->source = PyString_FromString("");
    if (self->source == NULL) {
      Py_DECREF(self);
      return NULL;
    }
    
    self->label = PyString_FromString("No data");
    if (self->label == NULL) {
      Py_DECREF(self);
      return NULL;
    }

    self->units = PyString_FromString("No units");
    if (self->units == NULL) {
      Py_DECREF(self);
      return NULL;
    }

    self->desc = PyString_FromString("No description");
    if (self->desc == NULL) {
      Py_DECREF(self);
      return NULL;
    }

    Py_INCREF(Py_None);
    self->dim = Py_None;
    
    self->order = 0;

    Py_INCREF(Py_None);
    self->time = Py_None;

    Py_INCREF(Py_None);
    self->errl = Py_None;
    Py_INCREF(Py_None);
    self->errh = Py_None;

    Py_INCREF(Py_None);
    self->data = Py_None;
  }
  
  return (PyObject *)self;
}

/* Initialise */
static int
Data_init(idam_Data *self, PyObject *args, PyObject *kwds)
{
  const char *data, *source;
  const char *host = NULL;
  char tmphost[MAXNAME];
  int port = -1, tmpport=-1;
  PyObject *source_obj;
  PyObject *tmp, *tmp2;

  idam_Dimension *dim;

  int handle;
  PyArrayObject* pyarr;
  int data_n;
  int rank, order;
  npy_intp dimsize[8];
  int i;

  static char *kwlist[] = {"data", "source", "host", "port", NULL};

  
  /* First argument is a string, second an object */
  if (! PyArg_ParseTupleAndKeywords(args, kwds, "sO|si", kwlist, 
				    &data, &tmp,
				    &host, &port))
    return -1; 

  /* Convert second argument to a string */
  source_obj = PyObject_Str(tmp); /* NB: This object is returned */
  source = PyString_AsString(source_obj); /* Refers to internal buffer */
  
  /* Check if an error occurred */
  if (source == NULL) {
    Py_XDECREF(self);
    PyErr_SetString(PyExc_RuntimeError, "Invalid arguments to idam.Data()");
    return -1;
  }

  /* Set host and port. Keep old settings for restore after */
  if(port > 0) {  
    tmpport = getIdamServerPort();
    putIdamServerPort(port);
  }
  if(host != NULL) {
    strcpy(tmphost, getIdamServerHost());
    putIdamServerHost(host);
  }

  /* Open connection and get data */
  printf("Connecting to %s:%d\n", getIdamServerHost(), getIdamServerPort());
  printf("Reading '%s' from '%s'\n", data, source);
  handle = idamGetAPI(data, source);

  if(!getIdamSourceStatus(handle)) {
    fprintf(stderr, "IDAM error: %s\n", getIdamErrorMsg(handle));
    /* Restore host and port */
    if(port > 0)
      putIdamServerPort(tmpport);
    if(host != NULL)
      putIdamServerHost(tmphost);
    
    PyErr_SetString(PyExc_RuntimeError, getIdamErrorMsg(handle));
    return -1;
  }
 
  /* Set data name and source */
  tmp = self->name;
  self->name = PyString_FromString(data);
  Py_XDECREF(tmp);
  tmp = self->source;
  self->source = source_obj;
  Py_XDECREF(tmp);

  /* Set data label, units and description */
  tmp = self->label;
  self->label = PyString_FromString(getIdamDataLabel(handle));
  Py_XDECREF(tmp);
  tmp = self->units;
  self->units = PyString_FromString(getIdamDataUnits(handle));
  Py_XDECREF(tmp);
  tmp = self->desc;
  self->desc = PyString_FromString(getIdamDataDesc(handle));
  Py_XDECREF(tmp);

  /* Get the size of the data array */
  if((data_n = getIdamDataNum(handle)) <= 0) {
    /* Restore host and port */
    if(port > 0)
      putIdamServerPort(tmpport);
    if(host != NULL)
      putIdamServerHost(tmphost);

    Py_XDECREF(self);
    PyErr_SetString(PyExc_RuntimeError, getIdamErrorMsg(handle));
    return -1;
  }
  
  /* Get the rank and order of the data */
  rank = getIdamRank(handle);
  order = getIdamOrder(handle);
  
  /* NOTE: Order of the dimensions is reversed */
  order = rank - 1 - order;

  for(i=0;i<rank;i++) {
    dimsize[rank-1-i] = getIdamDimNum(handle, i); 
  }

  /* Set the data */
  tmp = self->data;
  pyarr = (PyArrayObject*) PyArray_SimpleNew(rank,dimsize,PyArray_FLOAT);
  if (pyarr == NULL) {
    if(port > 0)
      putIdamServerPort(tmpport);
    if(host != NULL)
      putIdamServerHost(tmphost);
    Py_XDECREF(self);
    PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array for data");
    return -1;
  }
  getIdamFloatData(handle, (float *)(pyarr->data));
  self->data = PyArray_Return(pyarr);
  Py_XDECREF(tmp);

  /* Get the data errors (low and high asymmetric) */
  if(getIdamErrorType(handle) != TYPE_UNKNOWN) {
    /* Got error data */
    
    tmp = self->errl;
    pyarr = (PyArrayObject*) PyArray_SimpleNew(rank,dimsize,PyArray_FLOAT);
    if (pyarr == NULL) {
      if(port > 0)
	putIdamServerPort(tmpport);
      if(host != NULL)
	putIdamServerHost(tmphost);
      Py_XDECREF(self);
      PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array for error array");
      return -1;
    }
    getIdamFloatAsymmetricError(handle, 0, (float *)(pyarr->data));
    self->errl = PyArray_Return(pyarr);
    Py_XDECREF(tmp);

    tmp = self->errh;
    if(!getIdamErrorAsymmetry(handle)) {
      /* Error is symmetric. Just point to the same data */
      self->errh = self->errl;
      Py_INCREF(self->errh);
    }else {
      /* Need separate array */
      pyarr = (PyArrayObject*) PyArray_SimpleNew(rank,dimsize,PyArray_FLOAT);
      if (pyarr == NULL) {
	if(port > 0)
	  putIdamServerPort(tmpport);
	if(host != NULL)
	  putIdamServerHost(tmphost);
	Py_XDECREF(self);
        PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array for error array");
	return -1;
      }
      getIdamFloatAsymmetricError(handle, 1, (float *)(pyarr->data));
      self->errh = PyArray_Return(pyarr);
    }
    Py_XDECREF(tmp);
  }else {
    /* No error data, so set to None */
    
    tmp = self->errl;
    Py_INCREF(Py_None);
    self->errl = Py_None;
    Py_XDECREF(tmp);
    
    tmp = self->errh;
    Py_INCREF(Py_None);
    self->errh = Py_None;
    Py_XDECREF(tmp);
  }

  /* Set the time dimension to null */
  tmp = self->time;
  Py_INCREF(Py_None);
  self->time = Py_None;
  Py_XDECREF(tmp);

  /* Get the dimensions */

  tmp = self->dim;
  self->dim = PyList_New(rank);
  if(!(self->dim)) {
    Py_XDECREF(self);
    PyErr_SetString(PyExc_RuntimeError, "Could not create list of dimensions");
    return -1;
  }

  for(i=0;i<rank;i++) {
    dim = (idam_Dimension *) Dimension_new(&idam_DimensionType, NULL, NULL);
    
    tmp2 = dim->label;
    dim->label = PyString_FromString(getIdamDimLabel(handle, rank-1-i));
    Py_XDECREF(tmp2);
    
    tmp2 = dim->units;
    dim->units = PyString_FromString(getIdamDimUnits(handle, rank-1-i));
    Py_XDECREF(tmp2);
    
    tmp2 = dim->data;
    pyarr = (PyArrayObject*) PyArray_SimpleNew(1,&(dimsize[i]),PyArray_FLOAT);
    if (pyarr == NULL) {
      Py_XDECREF(self);
      PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array for dimension");
      return -1;
    }
    getIdamFloatDimData(handle, rank-1-i, (float *)(pyarr->data));
    dim->data = PyArray_Return(pyarr);
    Py_XDECREF(tmp2);
    
    if(getIdamDimErrorType(handle, rank-1-i) != TYPE_UNKNOWN) {
      tmp2 = dim->errl;
      pyarr = (PyArrayObject*) PyArray_SimpleNew(1,&(dimsize[i]),PyArray_FLOAT);
      if (pyarr == NULL) {
	Py_XDECREF(self);
        PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array for dimension error");
	return -1;
      }
      getIdamFloatDimAsymmetricError(handle, rank-1-i, 0, (float *)(pyarr->data));
      dim->errl = PyArray_Return(pyarr);
      Py_XDECREF(tmp2);
      
      tmp2 = dim->errh;
      if(!getIdamDimErrorAsymmetry(handle, rank-1-i)) {
	/* Symmetric error */
	dim->errh = dim->errl;
	Py_INCREF(self->errh);
      }else {
	/* Asymmetric error */
	pyarr = (PyArrayObject*) PyArray_SimpleNew(1,&(dimsize[i]),PyArray_FLOAT);
	if (pyarr == NULL) {
	  Py_XDECREF(self);
          PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array for dimension error");
	  return -1;
	}
	getIdamFloatDimAsymmetricError(handle, rank-1-i, 1, (float *)(pyarr->data));
	dim->errh = PyArray_Return(pyarr);
      }
      Py_XDECREF(tmp2);
    }else {
      /* No error data, so set to None */
      tmp2 = dim->errl;
      Py_INCREF(Py_None);
      dim->errl = Py_None;
      Py_XDECREF(tmp2);
      
      tmp2 = dim->errh;
      Py_INCREF(Py_None);
      dim->errh = Py_None;
      Py_XDECREF(tmp2);
    }
    
    /* Create shortcut to time data */
    if(i == order) { /* This is the time dimension */
      tmp2 = self->time;
      Py_INCREF(dim->data);
      self->time = dim->data;
      Py_XDECREF(tmp2);
    }
    
    /* Add this dimension to the list */
    PyList_SET_ITEM(self->dim, i, (PyObject*) dim);
  }
  Py_XDECREF(tmp); /* Delete the old dim list */
  
  /*  Set index of time dimension */
  self->order = order;
  
  /* Free IDAM data */
  idamFree(handle);
  
  /* Restore host and port */
  if(port > 0)
    putIdamServerPort(tmpport);
  if(host != NULL)
    putIdamServerHost(tmphost);

  return 0;
}

/* Methods */
static PyMethodDef idam_DataMethods[] = {
    {NULL}  /* Sentinel */
};

static PyTypeObject idam_DataType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "idam.Data",               /*tp_name*/
    sizeof(idam_Data),         /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Data_dealloc,  /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "IDAM data objects",       /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    idam_DataMethods,          /* tp_methods */
    idam_DataMembers,          /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Data_init,       /* tp_init */
    0,                         /* tp_alloc */
    Data_new,                  /* tp_new */
};

/************************************************************
 * Table of methods
 ************************************************************/

static PyMethodDef IdamMethods[] = {
  {"test",  idam_test, METH_VARARGS,
   "IDAM test code"},

  {"setHost",  idam_setHost, METH_VARARGS,
   "Set the host name of the IDAM server"},

  {"setPort",  idam_setPort, METH_VARARGS,
   "Set the port number of the IDAM server"},

  {"setProperty",  idam_setProperty, METH_VARARGS,
   "Set a property for client/server behavior"},

  {"getProperty",  idam_getProperty, METH_VARARGS,
   "Get a property for client/server behavior"},

  {"getAPI",  idam_getAPI, METH_VARARGS,
   "Low-level routine to open a connection"},

  {"freeAPI",  idam_freeAPI, METH_VARARGS,
   "Low-level routine to free a connection"},

  {"readData",  idam_readData, METH_VARARGS,
   "Low-level read a data array"},

  {NULL, NULL, 0, NULL}        /* Sentinel */
};

/************************************************************
 * Module initialisation
 ************************************************************/

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
initidam(void)
{
  PyObject *m;
  
  idam_DataType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&idam_DataType) < 0)
    return;

  idam_DimensionType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&idam_DimensionType) < 0)
    return;

  /* Initialise module */
  m = Py_InitModule3("idam", IdamMethods, "IDAM data access module.");
  
  if(m == NULL)
    return;

  /* Add types */
  Py_INCREF(&idam_DataType);
  PyModule_AddObject(m, "Data", (PyObject *)&idam_DataType);
  
  Py_INCREF(&idam_DimensionType);
  PyModule_AddObject(m, "Dimension", (PyObject *)&idam_DimensionType);
  
  /* Import NumPy */
  import_array();

  /* Initialise IDAM with default values */
  putIdamServerHost("mast.fusion.org.uk");
  putIdamServerPort(56565);

  /* Check for errors */
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module idam");
}
