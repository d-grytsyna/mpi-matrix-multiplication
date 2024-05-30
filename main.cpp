#include <mpi.h>
#include <iostream>
#define NRA 1500
#define NCA 1500
#define NCB 1500
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

double **alloc_2d(int rows, int cols) {
  /*allocate a continuous chunk*/
  double *m = (double *)malloc(rows * cols * sizeof(double));
  double **A = (double **)malloc(rows * sizeof(double *));

  A[0] = m;
  /*manually split it into rows*/
  for (int i = 1; i < rows; i++) A[i] = A[i - 1] + cols;

  return A;
}
void matrix_multiplication_blocking_workers(int argc, char *argv[]){
    int numtasks,
    taskid,
    numworkers,
    source,
    dest,
    rows, averow, extra, offset,
    i, j, k, rc;
    double **a = alloc_2d(NRA, NCA);
    double **b = alloc_2d(NCA, NCB);
    double **c = alloc_2d(NRA, NCB);

    MPI_Status status;
    MPI_Init( &argc, &argv);
    
    MPI_Comm_size( MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank( MPI_COMM_WORLD, &taskid);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD,rc);
        exit(1);
    }
    numworkers = numtasks-1;
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= 10;
        
        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                       b[i][j]= 10;
        
        double t1 = MPI_Wtime();
        averow = NRA/numworkers;
        extra = NRA%numworkers;
        offset = 0;
        for (dest=1; dest<=numworkers; dest++) {
            rows = (dest <= extra) ? averow+1 : averow;
            printf("Sending %d rows to task %d offset= %d\n", rows,dest, offset);
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER,MPI_COMM_WORLD);
            MPI_Send(a[offset], rows*NCA, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(b[0], NCA*NCB, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        for (source=1; source<=numworkers; source++) {
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(c[offset], rows*NCB, MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &status);
        }
        t1 = MPI_Wtime() - t1;
        printf("\nExecution time: %.2f\n", t1);
        printf("****\n");
    }else{
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(a[0], rows*NCA, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(b[0], NCA*NCB, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        for (k=0; k<NCB; k++)
            for (i=0; i<rows; i++) {
                      c[i][k] = 0.0;
                      for (j=0; j<NCA; j++)
                                c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(c[0], rows*NCB, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}

void matrix_multiplication_blocking(int argc, char *argv[]){
    int numtasks,
    taskid,
    numworkers,
    source,
    dest,
    rows, averow, extra, offset,
    i, j, k, rc;
    double **a = alloc_2d(NRA, NCA);
    double **b = alloc_2d(NCA, NCB);
    double **c = alloc_2d(NRA, NCB);


    MPI_Status status;
    MPI_Init( &argc, &argv);
    
    MPI_Comm_size( MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank( MPI_COMM_WORLD, &taskid);

//    if (numtasks < 2 ) {
//        printf("Need at least two MPI tasks. Quitting...\n");
//        MPI_Abort(MPI_COMM_WORLD,rc);
//        exit(1);
//    }
    numworkers = numtasks-1;
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= 10;
        
        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                       b[i][j]= 10;
        
        double t1 = MPI_Wtime();
        averow = NRA/numtasks;
        extra = NRA%numtasks;
        rows = (1 <= extra) ? averow+1 : averow;
        offset = rows;
        int master_rows = rows;
        for (dest=1; dest<=numworkers; dest++) {
            rows = (dest+1 <= extra) ? averow+1 : averow;
            printf("Sending %d rows to task %d offset= %d\n", rows,dest,offset);
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER,MPI_COMM_WORLD);
            MPI_Send(a[offset], rows*NCA, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(b[0], NCA*NCB, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        for (k=0; k<NCB; k++)
            for (i=0; i<master_rows; i++) {
                      c[i][k] = 0.0;
                      for (j=0; j<NCA; j++)
                                c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }
        
        for (source=1; source<=numworkers; source++) {
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(c[offset], rows*NCB, MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &status);
        }
        t1 = MPI_Wtime() - t1;
        printf("\nExecution time: %.2f\n", t1);
        printf("****\n");
    }else{
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(a[0], rows*NCA, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(b[0], NCA*NCB, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        for (k=0; k<NCB; k++)
            for (i=0; i<rows; i++) {
                      c[i][k] = 0.0;
                      for (j=0; j<NCA; j++)
                                c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(c[0], rows*NCB, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}

void matrix_multiplication_nonblocking_workers(int argc, char *argv[]){
    int numtasks,
    taskid,
    numworkers,
    source,
    dest,
    rows, averow, extra, offset,
    i, j, k, rc;
    double **a = alloc_2d(NRA, NCA);
    double **b = alloc_2d(NCA, NCB);
    double **c = alloc_2d(NRA, NCB);


    MPI_Status status;
    MPI_Init( &argc, &argv);
    
    MPI_Comm_size( MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank( MPI_COMM_WORLD, &taskid);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD,rc);
        exit(1);
    }
    numworkers = numtasks-1;
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= 10;
        
        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                       b[i][j]= 10;
        
        double t1 = MPI_Wtime();
        averow = NRA/numworkers;
        extra = NRA%numworkers;
        rows = (1 <= extra) ? averow+1 : averow;
        offset = 0;
        MPI_Request send_req[numworkers * 4];
        int offsets_arr[numworkers];
        int rows_arr[numworkers];
        for (dest=1; dest<=numworkers; dest++) {
            rows = (dest <= extra) ? averow+1 : averow;
            
            MPI_Isend(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD,&send_req[(dest - 1) * 4]);
            offsets_arr[dest-1] = offset;
            MPI_Isend(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD, &send_req[(dest - 1) * 4 + 1 ]);
            rows_arr[dest-1] = rows;
            MPI_Isend(a[offset], rows*NCA, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD, &send_req[(dest - 1) * 4 + 2 ]);
            MPI_Isend(b[0], NCA*NCB, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD, &send_req[(dest - 1) * 4 + 3 ]);

            offset = offset + rows;
        }
        
        MPI_Request recv_req2[numworkers];
        MPI_Status work_status[numworkers];
        for (source=1; source<=numworkers; source++) {
            MPI_Irecv(c[offsets_arr[source - 1]], rows_arr[source - 1] * NCB,
                      MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD,
                      &recv_req2[source - 1]);
        }
        MPI_Waitall(numworkers, &recv_req2[0], &work_status[0]);
        t1 = MPI_Wtime() - t1;
        printf("\nExecution time: %.2f\n", t1);
        printf("****\n");
    }else{
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(a[0], rows*NCA, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(b[0], NCA*NCB, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        for (k=0; k<NCB; k++)
            for (i=0; i<rows; i++) {
                      c[i][k] = 0.0;
                      for (j=0; j<NCA; j++)
                                c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }
        MPI_Send(c[0], rows*NCB, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}

void matrix_multiplication_nonblocking(int argc, char *argv[]){
    int numtasks,
    taskid,
    numworkers,
    source,
    dest,
    rows, averow, extra, offset,
    i, j, k, rc;
    double **a = alloc_2d(NRA, NCA);
    double **b = alloc_2d(NCA, NCB);
    double **c = alloc_2d(NRA, NCB);

    
    MPI_Status status;
    MPI_Init( &argc, &argv);
    
    MPI_Comm_size( MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank( MPI_COMM_WORLD, &taskid);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD,rc);
        exit(1);
    }
    numworkers = numtasks-1;
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= 10;
        
        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                b[i][j]= 10;
        
        double t1 = MPI_Wtime();
        averow = NRA/numtasks;
        extra = NRA%numtasks;
        rows = (1 <= extra) ? averow+1 : averow;
        offset = rows;
        int master_rows = rows;
        MPI_Status  recv_status1[numworkers * 2], recv_status2[numworkers];
        MPI_Request send_req[numworkers * 3], b_send_req[numworkers], recv_req1[numworkers * 2],
                recv_req2[numworkers];
        for (dest=1; dest<=numworkers; dest++) {
            rows = (dest+1 <= extra) ? averow+1 : averow;
            printf("Sending %d rows to task %d offset= %d\n", rows,dest,offset);
            MPI_Isend(b[0], NCA*NCB, MPI_DOUBLE, dest, FROM_MASTER+3, MPI_COMM_WORLD, &b_send_req[dest - 1]);
            
            MPI_Isend(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD, &send_req[(dest - 1) * 3]);
            MPI_Isend(&rows, 1, MPI_INT, dest, FROM_MASTER+1,MPI_COMM_WORLD, &send_req[(dest - 1) * 3 + 1]);
            
            MPI_Isend(a[offset], rows*NCA, MPI_DOUBLE, dest, FROM_MASTER + 2, MPI_COMM_WORLD, &send_req[(dest - 1) * 3 + 2]);
            offset = offset + rows;
        }
        MPI_Waitall(numworkers, b_send_req, MPI_STATUSES_IGNORE);
   

        for (k=0; k<NCB; k++)
            for (i=0; i<master_rows; i++) {
                      c[i][k] = 0.0;
                      for (j=0; j<NCA; j++)
                                c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }
            int offsets_arr[numworkers];
            int rows_arr[numworkers];

            for (source = 1; source <= numworkers; source++) {
              MPI_Irecv(&offsets_arr[source - 1], 1, MPI_INT, source, FROM_WORKER,
                        MPI_COMM_WORLD, &recv_req1[(source - 1) * 2]);
              MPI_Irecv(&rows_arr[source - 1], 1, MPI_INT, source, FROM_WORKER+1,
                        MPI_COMM_WORLD, &recv_req1[(source - 1) * 2 + 1]);
            }

        
            MPI_Waitall(numworkers * 2, &recv_req1[0], &recv_status1[0]);
            for (source = 1; source <= numworkers; source++) {
             MPI_Irecv(c[offsets_arr[source - 1]], rows_arr[source - 1] * NCB,
                       MPI_DOUBLE, source, FROM_WORKER+2, MPI_COMM_WORLD,
                       &recv_req2[source - 1]);
//             printf("Received results from task %d\n", source);
           }

           MPI_Waitall(numworkers, &recv_req2[0], &recv_status2[0]);

            t1 = MPI_Wtime() - t1;
            printf("\nExecution time: %.2f\n", t1);
            printf("****\n");
            } else { /* if (taskid > MASTER) */

        MPI_Status recv_status[4];
        MPI_Request send_req[3], recv_req[4];

        MPI_Irecv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD,
                  &recv_req[0]);
        MPI_Irecv(&rows, 1, MPI_INT, MASTER, FROM_MASTER+1, MPI_COMM_WORLD,
                  &recv_req[1]);
        MPI_Irecv(b[0], NCA * NCB, MPI_DOUBLE, MASTER, FROM_MASTER+3,
                  MPI_COMM_WORLD, &recv_req[3]);

        MPI_Wait(&recv_req[0], &recv_status[0]);
        MPI_Wait(&recv_req[1], &recv_status[1]);

        MPI_Isend(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD,
                  &send_req[0]);
        MPI_Isend(&rows, 1, MPI_INT, MASTER, FROM_WORKER+1, MPI_COMM_WORLD,
                  &send_req[1]);

        MPI_Irecv(a[0], rows * NCA, MPI_DOUBLE, MASTER, FROM_MASTER + 2,
                      MPI_COMM_WORLD, &recv_req[2]);


        MPI_Wait(&recv_req[2], &recv_status[2]);
        MPI_Wait(&recv_req[3], &recv_status[3]);

        for (k=0; k<NCB; k++)
            for (i=0; i<rows; i++) {
                      c[i][k] = 0.0;
                      for (j=0; j<NCA; j++)
                                c[i][k] = c[i][k] + a[i][j] * b[j][k];
            }
        

        MPI_Isend(c[0], rows * NCB, MPI_DOUBLE, MASTER, FROM_WORKER+2, MPI_COMM_WORLD, &send_req[2]);
        MPI_Status st;
        MPI_Wait(&send_req[2], &st);
      }
    MPI_Finalize();
    
    

}
int main(int argc, char *argv[]) {
    matrix_multiplication_blocking(argc, argv);
//    matrix_multiplication_nonblocking(argc, argv);
//    matrix_multiplication_blocking_workers(argc, argv);
//    matrix_multiplication_nonblocking_workers(argc, argv);
    return 0;
}
