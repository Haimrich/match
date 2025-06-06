/*
 * pulp_nn_conv_Co_parallel.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef CLUSTER_COMPILATION

#include <pulp.h>
#include <pulp_nn/pulp_nn_utils.h>
#include <pulp_nn/pulp_nn_kernels.h>

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)

void pulp_nn_conv_Co_parallel(
                              const uint8_t * pInBuffer,
                              uint8_t *       pIm2ColBuffer,
                              const int8_t *  bias,
                              uint8_t *       pOutBuffer,
                              const int8_t *  pWeight,
                              int32_t *       k,
                              int32_t *       lambda,
                              const uint16_t  out_mult,
                              const uint16_t  out_shift,
                              const uint16_t  dim_in_x,
                              const uint16_t  dim_in_y,
                              const uint16_t  ch_in,
                              const uint16_t  dim_out_x,
                              const uint16_t  dim_out_y,
                              const uint16_t  ch_out,
                              const uint16_t  dim_kernel_x,
                              const uint16_t  dim_kernel_y,
                              const uint16_t  padding_y_top,
                              const uint16_t  padding_y_bottom,
                              const uint16_t  padding_x_left,
                              const uint16_t  padding_x_right,
                              const uint16_t  stride_x,
                              const uint16_t  stride_y,
                              int             flag_relu,
                              int             flag_batch_norm)
{
  int core_id = rt_core_id();

  // local vars
  int i_out_y, i_out_x, i_ker_y, i_ker_x;

  int Log2Core = log2(get_core_num());
  /*chunks are built along the spatial dimension of the OFM */
  int chunk = (ch_out >> Log2Core) + ((ch_out & (get_core_num() - 1)) != 0);

  /* defining the specific channels computed by each core */
  int start_channel, stop_channel;
  start_channel = min(chunk * core_id, ch_out);
  stop_channel = min(start_channel + chunk, ch_out);

  int eff_chunk = stop_channel - start_channel;

  uint8_t *pIm2ColBase = pIm2ColBuffer + (2 * core_id * ch_in * dim_kernel_x * dim_kernel_y);
  uint8_t *pIm2Col = pIm2ColBase;
  int8_t *pW = pWeight + (start_channel * ch_in * dim_kernel_x * dim_kernel_y);

  uint8_t *pOut = pOutBuffer + start_channel;
  int32_t *k0 = k + start_channel;
  int32_t *lambda0 = lambda + start_channel;

  if(eff_chunk)
  {
    for (i_out_y = 0; i_out_y < dim_out_y; i_out_y++)
    {
      for (i_out_x = 0; i_out_x < dim_out_x; i_out_x++)
      {
        if(i_out_y < padding_y_top)
        {
          /* This part implements the im2col function */
          for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
          {
            for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
            {
              if (i_ker_y < 0 || i_ker_y >= dim_in_y || i_ker_x < 0 || i_ker_x >= dim_in_x)
              {
                pulp_zero_mem(pIm2Col, ch_in);
              }
              else
              {
                pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
              }
              pIm2Col += ch_in;
            }
          }
        }
        else if(i_out_y < dim_out_y - padding_y_bottom)
        {
          if(i_out_x < padding_x_left)
          {
            for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
            {
              for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
              {
                if (i_ker_x < 0 || i_ker_x >= dim_in_x)
                {
                  pulp_zero_mem(pIm2Col, ch_in);
                }
                else
                {
                  pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                }
                pIm2Col += ch_in;
              }
            }
          }
          else if(i_out_x < dim_out_x - padding_x_right)
          {
            for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
            {
              pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_y * dim_in_x + i_out_x * stride_x - padding_x_left) * ch_in, pIm2Col, ch_in * dim_kernel_x);
              pIm2Col += ch_in * dim_kernel_x;
            }
          }
          else
          {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
            {
              for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
                {
                  if (i_ker_x < 0 || i_ker_x >= dim_in_x)
                  {
                    pulp_zero_mem(pIm2Col, ch_in);
                  }
                  else
                  {
                    pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
                  }
                  pIm2Col += ch_in;
                }
            }
          }
        }
        else
        {
          for (i_ker_y = i_out_y * stride_y - padding_y_top; i_ker_y < i_out_y * stride_y - padding_y_top + dim_kernel_y; i_ker_y++)
          {
            for (i_ker_x = i_out_x * stride_x - padding_x_left; i_ker_x < i_out_x * stride_x - padding_x_left + dim_kernel_x; i_ker_x++)
            {
              if (i_ker_y < 0 || i_ker_y >= dim_in_y || i_ker_x < 0 || i_ker_x >= dim_in_x)
              {
                pulp_zero_mem(pIm2Col, ch_in);
              }
              else
              {
                pulp_nn_im2col_int8((uint8_t *) pInBuffer + (i_ker_y * dim_in_x + i_ker_x) * ch_in, pIm2Col, ch_in);
              }
              pIm2Col += ch_in;
            }
          }
        }
        if (pIm2Col == pIm2ColBase + 2 * ch_in * dim_kernel_x * dim_kernel_y)
        {
          pOut = ((ch_out - eff_chunk) << 1) + pulp_nn_matmul(
            pW,
            pIm2ColBase,
            eff_chunk,
            ch_in * dim_kernel_x * dim_kernel_y,
            out_shift,
            out_mult,
            k0,
            lambda0,
            bias,
            pOut,
            pOut + ch_out,
            flag_relu,
            flag_batch_norm
          );
          pIm2Col = pIm2ColBase;
        }
      }
    }

    /* check if there is left-over for compute */
    if (pIm2Col != pIm2ColBase)
    {
      const int8_t *pA = pW;
      int i;
      for (i = start_channel; i < stop_channel; i++)
      {
        int sum = 0;

        if (bias != NULL)
        {
          sum = ((int)(bias[i]));
        }

        uint8_t *pB = pIm2ColBase;
        /* basically each time it process 4 entries */
        uint16_t  col_cnt_im2col = ch_in * dim_kernel_x * dim_kernel_y >> 2;

        for (int j=0 ; j < col_cnt_im2col; j++)
        {
          v4s inA = *((v4s*) pA);
          v4u inB = *((v4u*) pB);

          sum = SumDotp(inB, inA, sum);
          pA+=4;
          pB+=4;
        }
        col_cnt_im2col = (ch_in * dim_kernel_y * dim_kernel_x) & 0x3;
        while (col_cnt_im2col)
        {
          int8_t      inA1 = *pA++;
          uint8_t     inB1 = *pB++;
          asm volatile("": : :"memory");
          sum += inA1 * inB1;

          col_cnt_im2col--;
        }
        /* if activation layer follows batch normalization */
        if (flag_batch_norm && flag_relu)
        {
          *pOut = pulp_nn_bn_quant_u8(sum, *k0, *lambda0, out_shift);
          k0++;
          lambda0++;
          pOut++;
        }
        else
        {
          /* if there isn't batch normalization but there is activation layer */
          if(flag_relu == 1)
          {
            *pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
          }
          else
          {
            *pOut = (uint8_t) clip8(sum >> out_shift);
          }
          pOut++;
        }
      }
    }
  }
}

#endif