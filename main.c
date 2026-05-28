#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // Librería agregada para medir el tiempo

#define WIDTH 7680
#define HEIGHT 4320
#define MAX_ITER 256

void generar_mandelbrot(unsigned char *imagen) {
    printf("Iniciando Tarea A: Generando Mandelbrot 8K...\n");
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

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
    printf("Iniciando Tarea B: Aplicando filtro Sobel...\n");
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

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

int main() {
    unsigned char *img_original = (unsigned char *)malloc(WIDTH * HEIGHT);
    unsigned char *img_filtrada = (unsigned char *)calloc(WIDTH * HEIGHT, sizeof(unsigned char));

    if (!img_original || !img_filtrada) {
        printf("Error: No hay suficiente memoria para imagenes 8K.\n");
        return 1;
    }

    struct timespec start, end;
    double tiempo_total;

    printf("========================================\n");
    printf("Iniciando procesamiento secuencial...\n");

    // Iniciar el cronómetro
    clock_gettime(CLOCK_MONOTONIC, &start);

    generar_mandelbrot(img_original);
    aplicar_filtro_sobel(img_original, img_filtrada);

    // Detener el cronómetro
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calcular el tiempo transcurrido en segundos
    tiempo_total = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("========================================\n");
    printf("TIEMPO TOTAL DE EJECUCION: %.6f segundos\n", tiempo_total);
    printf("========================================\n");

    guardar_imagen("mandelbrot_sobel_8k.ppm", img_filtrada);

    free(img_original);
    free(img_filtrada);

    return 0;
}
