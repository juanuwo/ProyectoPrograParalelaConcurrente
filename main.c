#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Librería de OpenMP agregada

#define WIDTH 7680
#define HEIGHT 4320
#define MAX_ITER 256

void generar_mandelbrot(unsigned char *imagen) {
    printf("Iniciando Tarea A: Generando Mandelbrot 8K (Paralelo)...\n");
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

    // Directiva de OpenMP para paralelizar el ciclo externo
    #pragma omp parallel for schedule(guided, 32)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double c_re = x_min + (x * (x_max - x_min) / WIDTH);
            double c_im = y_min + (y * (y_max - y_min) / HEIGHT);
            double z_re = 0, z_im = 0;
            int iter = 0;

            while (z_re * z_re + z_im * z_im <= 4.0 && iter < MAX_ITER) {
                double z_re_new = z_re * z_re - z_im * z_im + c_re;
                z_im = 2.0 * z_re * z_im + c_im;
                z_re = z_re_new;
                iter++;
            }
            imagen[y * WIDTH + x] = (iter == MAX_ITER) ? 0 : (iter * 255 / MAX_ITER);
        }
    }
    printf("Mandelbrot generado.\n");
}

void aplicar_filtro_sobel(unsigned char *imagen_in, unsigned char *imagen_out) {
    printf("Iniciando Tarea B: Aplicando filtro Sobel (Paralelo)...\n");
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Directiva de OpenMP para paralelizar el ciclo externo
    #pragma omp parallel for
    for (int y = 1; y < HEIGHT - 1; y++) {
        for (int x = 1; x < WIDTH - 1; x++) {
            int sum_x = 0;
            int sum_y = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel_val = imagen_in[(y + ky) * WIDTH + (x + kx)];
                    sum_x += pixel_val * Gx[ky + 1][kx + 1];
                    sum_y += pixel_val * Gy[ky + 1][kx + 1];
                }
            }

            int magnitud = (int)sqrt((double)(sum_x * sum_x + sum_y * sum_y));
            if (magnitud > 255) magnitud = 255;
            if (magnitud < 0) magnitud = 0;

            imagen_out[y * WIDTH + x] = (unsigned char)magnitud;
        }
    }
    printf("Filtro Sobel aplicado.\n");
}

void guardar_imagen(const char *filename, unsigned char *imagen) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error al abrir el archivo para guardar");
        exit(1);
    }
    fprintf(fp, "P5\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(imagen, 1, WIDTH * HEIGHT, fp);
    fclose(fp);
    printf("Imagen guardada como %s\n", filename);
}

// Implementación 1: Exclusión Mutua (Lenta y con alta contención)
void histograma_atomico(unsigned char *imagen) {
    int histograma[256] = {0};
    printf("Calculando histograma (Atómico / Exclusión Mutua)...\n");

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        int color = imagen[i];

        // Exclusión mutua: Obliga a los hilos a formarse para escribir
        #pragma omp atomic
        histograma[color]++;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double tiempo = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("-> Tiempo Histograma Atómico: %.6f segundos\n", tiempo);
}

// Implementación 2: Variables Locales / Reduction (Rápida)
void histograma_reduccion(unsigned char *imagen) {
    int histograma[256] = {0};
    printf("Calculando histograma (Reduction / Variables Locales)...\n");

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Cada hilo crea su propia copia local del arreglo y al final los suma
    #pragma omp parallel for reduction(+:histograma) schedule(static)
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        histograma[imagen[i]]++;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double tiempo = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("-> Tiempo Histograma Reduction: %.6f segundos\n", tiempo);
}

int main() {
    unsigned char *img_original = (unsigned char *)malloc(WIDTH * HEIGHT);
    unsigned char *img_filtrada = (unsigned char *)calloc(WIDTH * HEIGHT, sizeof(unsigned char));

    if (!img_original || !img_filtrada) {
        printf("Error: No hay suficiente memoria para imagenes 8K.\n");
        return 1;
    }

    struct timespec start_A, end_A, start_B, end_B;
    double tiempo_A, tiempo_B;

    printf("========================================\n");
    printf("Iniciando procesamiento con Schedulers...\n");

    // Medir solo Tarea A
    clock_gettime(CLOCK_MONOTONIC, &start_A);
    generar_mandelbrot(img_original);
    clock_gettime(CLOCK_MONOTONIC, &end_A);
    tiempo_A = (end_A.tv_sec - start_A.tv_sec) + (end_A.tv_nsec - start_A.tv_nsec) / 1e9;

    // Medir solo Tarea B
    clock_gettime(CLOCK_MONOTONIC, &start_B);
    aplicar_filtro_sobel(img_original, img_filtrada);
    clock_gettime(CLOCK_MONOTONIC, &end_B);
    tiempo_B = (end_B.tv_sec - start_B.tv_sec) + (end_B.tv_nsec - start_B.tv_nsec) / 1e9;

    printf("========================================\n");
    printf("TIEMPO TAREA A (Mandelbrot): %.6f segundos\n", tiempo_A);
    printf("TIEMPO TAREA B (Sobel):      %.6f segundos\n", tiempo_B);
    printf("TIEMPO TOTAL:                %.6f segundos\n", tiempo_A + tiempo_B);
    printf("========================================\n");

    printf("========================================\n");
    histograma_atomico(img_filtrada);
    histograma_reduccion(img_filtrada);
    printf("========================================\n");

    guardar_imagen("mandelbrot_sobel_8k.ppm", img_filtrada);

    free(img_original);
    free(img_filtrada);

    return 0;
}
