/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*****************
*  rtk #includes *
*****************/
#include "rtkConfiguration.h"
#include "rtkCudaBackProjectionImageFilter.hcu"
#include "rtkCudaUtilities.hcu"
#include "rtkMacro.h"

/*****************
*  C   #includes *
*****************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*****************
* CUDA #includes *
*****************/
#include <cuda.h>

// P R O T O T Y P E S ////////////////////////////////////////////////////
__global__ void kernel(float *dev_vol, int3 vol_dim, unsigned int Blocks_Y);

///////////////////////////////////////////////////////////////////////////

// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 2, cudaReadModeElementType> tex_img;
texture<float, 1, cudaReadModeElementType> tex_matrix;
///////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__
void kernel(float *dev_vol, int3 vol_dim, unsigned int Blocks_Y)
{
  // CUDA 2.0 does not allow for a 3D grid, which severely
  // limits the manipulation of large 3D arrays of data.  The
  // following code is a hack to bypass this implementation
  // limitation.
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*vol_dim.y)*(vol_dim.x);

  float3 ip;
  float  voxel_data;

  // matrix multiply
  ip.x = tex1Dfetch(tex_matrix, 0)*i + tex1Dfetch(tex_matrix, 1)*j +
         tex1Dfetch(tex_matrix, 2)*k + tex1Dfetch(tex_matrix, 3);
  ip.y = tex1Dfetch(tex_matrix, 4)*i + tex1Dfetch(tex_matrix, 5)*j +
         tex1Dfetch(tex_matrix, 6)*k + tex1Dfetch(tex_matrix, 7);
  ip.z = tex1Dfetch(tex_matrix, 8)*i + tex1Dfetch(tex_matrix, 9)*j +
         tex1Dfetch(tex_matrix, 10)*k + tex1Dfetch(tex_matrix, 11);

  // Change coordinate systems
  ip.z = 1 / ip.z;
  ip.x = ip.x * ip.z;
  ip.y = ip.y * ip.z;

  // Get texture point, clip left to GPU
  voxel_data = tex2D(tex_img, ip.x, ip.y);

  // Place it into the volume
  dev_vol[vol_idx] += voxel_data;
}

__global__
void kernel_optim(float *dev_vol, int3 vol_dim)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = 0;
  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= vol_dim.x || k >= vol_dim.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + k*vol_dim.y*vol_dim.x;

  float3 ip;

  // matrix multiply
  ip.x = tex1Dfetch(tex_matrix, 0)*i + tex1Dfetch(tex_matrix, 2)*k + tex1Dfetch(tex_matrix, 3);
  ip.y = tex1Dfetch(tex_matrix, 4)*i + tex1Dfetch(tex_matrix, 6)*k + tex1Dfetch(tex_matrix, 7);
  ip.z = tex1Dfetch(tex_matrix, 8)*i + tex1Dfetch(tex_matrix, 10)*k + tex1Dfetch(tex_matrix, 11);

  // Change coordinate systems
  ip.z = 1 / ip.z;
  ip.x = ip.x * ip.z;
  ip.y = ip.y * ip.z;
  float dx = tex1Dfetch(tex_matrix, 1)*ip.z;
  float dy = tex1Dfetch(tex_matrix, 5)*ip.z;

  ip.z*=ip.z;

  // Place it into the volume segment
  for(; j<vol_dim.y; j++)
    {
    dev_vol[vol_idx] += tex2D(tex_img, ip.x, ip.y);
    vol_idx+=vol_dim.x;
    ip.x+=dx;
    ip.y+=dy;
    }
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_back_project_init() /////////////////////////////
void
CUDA_back_project_init(
  int img_dim[2],
  int vol_dim[3],
  float *&dev_vol,         // Holds voxels on device
  float *&dev_img,         // Holds image pixels on device
  float *&dev_matrix,       // Holds matrix on device
  const float *host_vol
  )
{
  // Size of volume Malloc
  size_t vol_size_malloc = (vol_dim[0]*vol_dim[1]*vol_dim[2])*sizeof(float);

  // CUDA device pointers
  cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
  CUDA_CHECK_ERROR;
  cudaMalloc( (void**)&dev_vol, vol_size_malloc);
  CUDA_CHECK_ERROR;
  //cudaMemset( (void *) dev_vol, 0, vol_size_malloc);
  cudaMemcpy ((void *)dev_vol, host_vol, vol_size_malloc, cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMallocArray( (cudaArray **)&dev_img, &channelDesc, img_dim[0], img_dim[1] );

  // set texture parameters
  tex_img.addressMode[0] = cudaAddressModeClamp; //For the time being, clamp mode.
  tex_img.addressMode[1] = cudaAddressModeClamp; //For the time being, clamp mode.
  tex_img.filterMode = cudaFilterModeLinear;
  tex_img.normalized = false; // don't access with normalized texture coords

}

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_back_project() //////////////////////////////////
void
CUDA_back_project(
  int img_dim[2],
  int vol_dim[3],
  float *proj,
  float matrix[12],
  float *dev_vol,
  float *dev_img,
  float *dev_matrix
  )
  //const float *host_vol,
  //const float *host_output)
{

  // Size of volume Malloc
  //size_t vol_size_malloc = (vol_dim[0]*vol_dim[1]*vol_dim[2])*sizeof(float);
  //cudaMemcpy (dev_vol, host_vol, vol_size_malloc, cudaMemcpyHostToDevice);

  // copy image data, bind the array to the texture
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaMemcpyToArray( (cudaArray*)dev_img, 0, 0, proj, img_dim[0] * img_dim[1] * sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTextureToArray( tex_img, (cudaArray*)dev_img, channelDesc);

  // copy matrix, bind data to the texture
  cudaMemcpy (dev_matrix, matrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_matrix, dev_matrix, 12*sizeof(float) );

  // The optimized version runs when only one of the axis of the detector is
  // parallel to the y axis of the volume

  if(fabs(matrix[5])<1e-10 && fabs(matrix[9])<1e-10)
    {
    // Thread Block Dimensions
    static const int tBlock_x = 32;
    static const int tBlock_y = 16;

    // Each segment gets 1 thread
    int  blocksInX = vol_dim[0]/tBlock_x;
    int  blocksInY = vol_dim[2]/tBlock_y;
    dim3 dimGrid  = dim3(blocksInX, blocksInY);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, 1);
    // Note: cbi->img AND cbi->matrix are passed via texture memory
    //-------------------------------------
    kernel_optim <<< dimGrid, dimBlock >>> ( dev_vol, make_int3(vol_dim[0], vol_dim[1], vol_dim[2]) );
    }
  else
    {
    // Thread Block Dimensions
    static const int tBlock_x = 16;
    static const int tBlock_y = 4;
    static const int tBlock_z = 4;

    // Each element in the volume (each voxel) gets 1 thread
    int  blocksInX = (vol_dim[0]-1)/tBlock_x + 1;
    int  blocksInY = (vol_dim[1]-1)/tBlock_y + 1;
    int  blocksInZ = (vol_dim[2]-1)/tBlock_z + 1;
    dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    // Note: cbi->img AND cbi->matrix are passed via texture memory
    //-------------------------------------
    kernel <<< dimGrid, dimBlock >>> ( dev_vol,
                                       make_int3(vol_dim[0], vol_dim[1], vol_dim[2]),
                                       blocksInY );
    }
  CUDA_CHECK_ERROR;

  // Copy reconstructed volume from device to host
  //cudaMemcpy ((void *)host_output, dev_vol, vol_size_malloc, cudaMemcpyDeviceToHost);
  //CUDA_CHECK_ERROR;

}

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_back_project_cleanup() //////////////////////////
void
CUDA_back_project_cleanup(
  int vol_dim[3],
  float *dev_vol,
  float *dev_img,
  float *dev_matrix,
  const float *host_output
  )

{
  // Size of volume Malloc
  size_t vol_size_malloc = (vol_dim[0]*vol_dim[1]*vol_dim[2])*sizeof(float);
  // Copy reconstructed volume from device to host
  cudaMemcpy ((void *)host_output, dev_vol, vol_size_malloc, cudaMemcpyDeviceToHost);
  CUDA_CHECK_ERROR;

  // Unbind the image and projection matrix textures
  cudaUnbindTexture (tex_img);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_matrix);
  CUDA_CHECK_ERROR;

  // Cleanup
  cudaFreeArray ((cudaArray*)dev_img);
  CUDA_CHECK_ERROR;
  cudaFree (dev_matrix);
  CUDA_CHECK_ERROR;
  cudaFree (dev_vol);
  CUDA_CHECK_ERROR;
}
