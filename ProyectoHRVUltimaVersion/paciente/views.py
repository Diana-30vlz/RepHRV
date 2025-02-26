from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
from django.db.models import Q
import pandas as pd 
from datetime import datetime
from django.http import JsonResponse
import math

from weasyprint import HTML
from django.http import HttpResponse
from django.conf import settings

from django.template.loader import render_to_string

import numpy as np
from scipy.signal import firwin, lfilter, find_peaks, welch
from scipy.interpolate import interp1d
from pyhrv import time_domain
from io import BytesIO
import base64
import json
import plotly
from django.core.serializers.json import DjangoJSONEncoder
from biosppy.signals import ecg
from django.utils import timezone

from biosppy.signals.ecg import engzee_segmenter, correct_rpeaks
from django.views.decorators.csrf import csrf_exempt





def signup(request):
    if request.method == 'GET':
        departamentos = Departamento.objects.all()
        form = UserRegistrationForm()
        return render(request, 'signup.html', {"form": form, "departamentos": departamentos})
    else:
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            password1 = form.cleaned_data.get('password1')
            password2 = form.cleaned_data.get('password2')
            if password1 == password2:
                try:
                    # Intento de crear el usuario
                    user = form.save(commit=False)
                    user.set_password(password1)  # Encripta la contraseña antes de guardar el usuario
                    user.save()

                    # Recoge los datos del formulario adicionales
                    nombre_especialista = request.POST.get("nombres")
                    apellido_paterno = request.POST.get("apellido_paterno")
                    apellido_materno = request.POST.get("apellido_materno")
                    telefono = request.POST.get("telefono")
                    correo = request.POST.get("correo")
                    especialidad = request.POST.get("especialidad")
                    departamento_id = request.POST.get("departamento")
                    fecha_nacimiento = request.POST.get("fecha_nacimiento")

                    # Validación de que la fecha de nacimiento no sea nula
                    if not fecha_nacimiento:
                        return render(request, 'signup.html', {
                            "form": form,
                            "departamentos": Departamento.objects.all(),
                            "error": "Por favor, proporciona una fecha de nacimiento válida."
                        })

                    # Obtiene la instancia de Departamento usando el ID proporcionado
                    departamento = Departamento.objects.get(id_departamento=departamento_id)

                    # Crea el objeto Especialista
                    especialista = Especialista.objects.create(
                        user=user,
                        nombre_especialista=nombre_especialista,
                        apellido_paterno=apellido_paterno,
                        apellido_materno=apellido_materno,
                        telefono=telefono,
                        correo=correo,
                        especialidad=especialidad,
                        fecha_nacimiento=fecha_nacimiento,
                        departamento_id=departamento  # Ahora asigna la instancia de Departamento
                    )
                    especialista.save()

                    login(request, user)
                    return redirect('homeDoctor')
                except IntegrityError as e:
                    print(e)  # Esto imprimirá la excepción completa en la consola.
                    if 'UNIQUE constraint' in str(e):
                        error_message = "El usuario ya existe. Prueba con otro nombre de usuario."
                    else:
                        error_message = f"Ocurrió un error durante el registro: {e}."  # Muestra el error específico
                    return render(request, 'signup.html', {
                        "form": form,
                        "departamentos": Departamento.objects.all(),
                        "error": error_message
                    })

            else:
                # Si las contraseñas no coinciden
                return render(request, 'signup.html', {
                    "form": form,
                    "departamentos": Departamento.objects.all(),
                    "error": "Las contraseñas no coinciden."
                })
        
        # Si el formulario no es válido
        return render(request, 'signup.html', {
            "form": form,
            "departamentos": Departamento.objects.all(),
            "error": "Por favor corrige los errores del formulario."
        })

# Vista para mostrar la página de inicio
def home(request):
    return render(request, 'home.html')


# Vista para el inicio de sesión
def signin(request):
    
    if request.method == 'GET':
        return render(request, 'signin.html', {"form": AuthenticationForm()})
    else:
        user = authenticate(
            request, username=request.POST['username'], password=request.POST['password1'])
        if user is None:
            return render(request, 'signin.html', {"form": AuthenticationForm(), "error": "Nombre de usuario o contraseña incorrectos."})

        login(request, user)
        return redirect('homeDoctor')

# Vista para cerrar la sesión de un usuario
@login_required
def signout(request):
    logout(request)
    return redirect('home')  # Redirige a la URL 'home', que apunta a la vista 'home' definida en urls.py


# Vista para mostrar los pacientes pendientes (no completados)
@login_required
def pacientes(request):
    query = request.GET.get('query', '')  # Captura el parámetro de búsqueda desde la URL
    especialista = request.user.especialista  # Obtiene el especialista asociado al usuario actual

    if query:
        # Filtra pacientes del especialista actual que coincidan con el criterio de búsqueda
        pacientes = Paciente.objects.filter(
            Q(especialista=especialista) &
            (
                Q(nombre_paciente__icontains=query) |
                Q(apellido_paterno__icontains=query) |
                Q(apellido_materno__icontains=query) |
                Q(id_paciente__icontains=query) |
                Q(sexo__icontains=query) |
                Q(correo__icontains=query)
            )
        )
    else:
        # Si no hay búsqueda, muestra todos los pacientes del especialista actual
        pacientes = Paciente.objects.filter(especialista=especialista)

    return render(request, 'paciente.html', {"pacientes": pacientes})



def buscar_registro(request):
    registro_busqueda = request.GET.get('registro_busqueda', '')  # Captura el parámetro de búsqueda desde la URL
    paciente = request.user.Paciente  # Obtiene el especialista asociado al usuario actual

    if registro_busqueda:
        # Filtra pacientes del especialista actual que coincidan con el criterio de búsqueda
        registro = ECG.objects.filter(
                Q(registro=registro) &
            (
                Q(homoclave__icontains=registro_busqueda) |
                Q(fecha_informe__icontains=registro_busqueda) |
                Q(apellido_materno__icontains=registro_busqueda) 
            )
        )
    else:
        # Si no hay búsqueda, muestra todos los pacientes del especialista actual
        pacientes = Paciente.objects.filter(paciente=paciente)

    return render(request, 'paciente.html', {"pacientes": pacientes})

    


@login_required
def editar_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)  # Cambiado a id_paciente
    if request.method == 'POST':
        form = PacienteForm(request.POST, instance=paciente)
        if form.is_valid():
            form.save()
            return redirect('pacientes')
    else:
        form = PacienteForm(instance=paciente)
    return render(request, 'editar_paciente.html', {'form': form})

'''En la plantilla editar.html, puedes usar el formulario de la siguiente manera:
#{'Formulario': Formulario} significa que dentro de la plantillaeditar.html, puedes acceder a la instancia del formulario con la variable Formulario.'''


@login_required
def eliminar_paciente(request, paciente_id):
    pacientes = get_object_or_404(Paciente, id_paciente=paciente_id) # referencia al campo de la clase 
    pacientes.delete() # Elimina el paciente
    return redirect('pacientes') # Redirige a la lista de pacientes


@login_required
def historial(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)
    registros_ecg = ECG.objects.filter(paciente=paciente)  # Filtrar por el paciente específico
    
    # Verificación de si hay registros
    no_registros = registros_ecg.count() == 0
    
    # Depuración: mostrar en la consola si no hay registros
    print(f"No hay registros para el paciente {paciente_id}: {no_registros}")
    
    return render(request, 'historial.html', {'paciente': paciente, 'registros_ecg': registros_ecg, 'no_registros': no_registros})


@login_required
def buscar(request):
    query = request.GET.get('query', '')
    pacientes = Paciente.objects.filter(
        Q(nombre_paciente__icontains=query) |
        Q(apellido_paterno__icontains=query)
    ) if query else []
    
    return render(request, 'paciente.html', {'pacientes': pacientes, 'query': query})

# Vista para crear un nuevo paciente
@login_required
def create_paciente(request):
    if request.method == "GET":
        return render(request, 'create_paciente.html', {"form": PacienteForm()})
    else:
        form = PacienteForm(request.POST)
        if form.is_valid():
            try:
                # Buscar el especialista vinculado al usuario actual
                especialista = Especialista.objects.get(user=request.user)
                
                # Obtener los datos del formulario (sin usuario y contraseña)
                nombre_paciente = form.cleaned_data['nombre_paciente']
                apellido_paterno = form.cleaned_data['apellido_paterno']
                apellido_materno = form.cleaned_data['apellido_materno']
                telefono = form.cleaned_data['telefono']
                correo = form.cleaned_data['correo']
                sexo = form.cleaned_data['sexo']
                fecha_nacimiento = form.cleaned_data['fecha_nacimiento']

                # Crear el usuario para el paciente (sin usuario_paciente y contrasenia_paciente)
                user = User.objects.create_user(username=nombre_paciente, password="defaultpassword")  # Aquí puedes elegir una lógica para el password
                user.save()

                # Crear el paciente, asignando el usuario creado y el especialista actual
                new_paciente = form.save(commit=False)
                new_paciente.user = user  # Vincula el paciente con el usuario creado
                new_paciente.especialista = especialista  # Vincula el paciente con el especialista actual
                new_paciente.save()

                # Redirigir a la página de pacientes
                return redirect('pacientes')

            except Especialista.DoesNotExist:
                return render(request, 'create_paciente.html', {
                    "form": form,
                    "error": "El especialista no está registrado. Por favor, verifica tu cuenta."
                })
            except ValueError:
                return render(request, 'create_paciente.html', {
                    "form": form,
                    "error": "Surgió un error al crear al paciente."
                })
            except Exception as e:
                return render(request, 'create_paciente.html', {
                    "form": form,
                    "error": f"Hubo un error: {str(e)}"
                })
        
        # Si el formulario no es válido, se muestra el error
        print(form.errors)
        return render(request, 'create_paciente.html', {
            "form": form,
            "error": "Por favor corrige los errores del formulario."
        })

@login_required
def crear_informe(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)
    Formulario = ExpedienteForm(request.POST or None, request.FILES or None)  # Cargar los datos del formulario

    if request.method == 'POST':  # Solo manejamos el archivo cuando sea POST
        if Formulario.is_valid():
            # Validación adicional para el archivo
            archivo = request.FILES.get('archivo_ecg')
            if archivo:
                if not (archivo.name.endswith('.txt') or archivo.name.endswith('.csv')):
                    Formulario.add_error('archivo_ecg', 'El archivo debe ser de tipo .txt o .csv.')
                else:
                    expediente = Formulario.save(commit=False)  # No guarda todavía
                    expediente.paciente = paciente  # Relaciona el paciente
                    expediente.save()  # Ahora guarda el informe

                    # Leer el archivo solo si es válido
                    try:
                        ECG = pd.read_csv(archivo, sep='\s+', header=None)
                        if ECG.shape[1]==2:
                            tiempo = ECG.iloc[:,0]
                            voltaje = ECG.iloc[:,1]
                            fm = int(1/(tiempo.iloc[1]-tiempo.iloc[0]))
                            distacia = fm * 0.2


                           # fig = px.line(x=tiempo, y=voltaje, labels={'x': 'Tiempo (s)', 'y': 'Voltaje (mV)'}, title='ECG')
                            
                            #fig_path = f'media/graficos/ecg_{paciente.id_paciente}.html'
                            #fig.write_html(fig_path)
                    except Exception as e:
                        Formulario.add_error('archivo_ecg', f'Error al leer el archivo: {str(e)}')
                    return redirect('historial', paciente_id=paciente.id_paciente)
            else:
                Formulario.add_error('archivo_ecg', 'Debes seleccionar un archivo.')
        else:
            print(Formulario.errors)  # Mostrar los errores si los hay

    # Si es una solicitud GET o si hay errores en el formulario, renderizamos de nuevo el formulario
    return render(request, 'crear_informe.html', {'Formulario': Formulario, 'id_paciente': paciente.id_paciente})

@login_required
def eliminar_informe(request, paciente_id):
    ecg = get_object_or_404(ECG, id_ecg=paciente_id) # referencia al campo de la clase 
    ecg.delete() # Elimina el paciente
    return redirect('historial', paciente_id=ecg.paciente.id_paciente ) # Redirige a la lista de pacientes

def formato_ecg(file):
    delimitadores = [",", "\t", " ", ";"]
    for delim in delimitadores:
        try:
            datos_ecg = pd.read_csv(file, sep=delim)
            if datos_ecg.shape[1] >= 2:  # Verifica que haya al menos dos columnas
                return datos_ecg
        except pd.errors.EmptyDataError:
            print('El archivo está vacío.')
            return None
        except pd.errors.ParserError:
            print('Error de análisis al leer el archivo.')
            return None
        except Exception as e:
            print('No se puede leer el archivo: ', str(e))
    print(f'Formato de archivo no soportado: {file.name}')
    return None

def convertir_tipos(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convertir arrays a listas
    elif isinstance(obj, np.float64) or isinstance(obj, np.int64):
        return obj.item()  # Convertir números de numpy a tipos estándar de Python
    elif isinstance(obj, dict):
        return {k: convertir_tipos(v) for k, v in obj.items()}  # Convertir diccionarios recursivamente
    elif isinstance(obj, (tuple, list)):
        return [convertir_tipos(i) for i in obj]  # Convertir listas o tuplas recursivamente
    else:
        return obj  # Mantener los demás tipos igual

def filtro_pasa_bajas(v, fs):
    señal_corregida = v - np.mean(v)
    low_cutoff = 10
    high_cutoff = 30
    coeficientes = firwin(101, [low_cutoff, high_cutoff], fs=fs, window='hamming', pass_zero=False)
    V_FIL =  lfilter(coeficientes, 1.0, señal_corregida)

    return (V_FIL)

def algoritmo_segunda_derivada(v):#tiene entrada la señal filtrada
    coeficientes = np.array([1,1,1,1,-10,1,1,1,1])
    ECG_A = 4 * (np.convolve(v, coeficientes, mode= 'same' ))
    UM = 0
    return (ECG_A*ECG_A), UM

def detección_RR(V, T, FC):
    output = engzee_segmenter(signal= V, sampling_rate= FC, threshold= 0.6)
    rpeaks =  output['rpeaks']
    rpeaks_corr = correct_rpeaks(signal= V, rpeaks=rpeaks, sampling_rate= FC, tol = 0.05)
    rpeaks_corr = rpeaks_corr['rpeaks']
    intervalosRR = np.diff(T[rpeaks_corr])
    umbral_segundos = 300/1000
    picos_validos = [rpeaks_corr[0]]
    for i in range(1, len(rpeaks_corr)):
        if intervalosRR[i-1] >= umbral_segundos:
            picos_validos.append(rpeaks_corr[i])

    
    
    return rpeaks_corr, intervalosRR*1000

def Parametro_dominio_tiempo(intervalos_RR):
    resultadosDT = time_domain.time_domain(intervalos_RR)
    resultadosDT_obj = {
        "nni_mean": resultadosDT[1],
        "nni_min": resultadosDT[2],
        "nni_max": resultadosDT[3],
        "hr_mean": resultadosDT[7],
        "hr_min": resultadosDT[8],
        "hr_max": resultadosDT[9],
        "hr_std": resultadosDT[10],
        "nni_diff_mean": resultadosDT[4],
        "nni_diff_min": resultadosDT[5],
        "nni_diff_max": resultadosDT[6],
        "sdnn": resultadosDT[11],
        "sdann": resultadosDT[13],
        "rmssd": resultadosDT[14],
        "sdsd": resultadosDT[15],
        "nn50": resultadosDT[18],
        "pnn50": resultadosDT[18],
        "nn20": resultadosDT[18],
    }

    resultadosDT = tuple(convertir_tipos(i) for i in resultadosDT)
    print(type(resultadosDT))  # Debería decir <class 'tuple'>
    print(len(resultadosDT))   # Ver cuántos elementos tiene la tupla

    for i, item in enumerate(resultadosDT):
        print(f"Elemento {i}: Tipo {type(item)}")
    SDNN = 0


   # print(resultadosDT)
    return resultadosDT_obj, SDNN


def Parametro_dominio_frecuencia(RR, intervalos_RR, T):
    #Interpolar el Tacograma para Obtener un Muestreo Regular
    RR = np.array(RR)
    intervalos_RR = np.array(intervalos_RR)
    T = np.array(T)
    fs = 4
    T_RR = T[np.array(RR, dtype = int)]
    tiempo_regular = np.arange(T_RR[0],T[RR][-1], 1 /fs ) 
    interp_func = interp1d(T_RR,intervalos_RR, kind='cubic')
    rr_uniforme = interp_func(tiempo_regular)
    print(f'Se realizo la interpolación correctamente ')

    return fs, tiempo_regular, rr_uniforme

def calcular_potencia_banda(frecuencias, psd, banda):
    indices = np.where((frecuencias >= banda[0]) & (frecuencias <= banda[1]))[0]
    potencia = np.trapz(psd[indices], frecuencias[indices])
    frecuencia_pico = frecuencias[indices][np.argmax(psd[indices])] if indices.size > 0 else 0

    return potencia, frecuencia_pico

# Cálculo de parámetros de dominio de frecuencia usando Welch
def calculo_welch(RR, FS):
    # Estimación PSD con el método de Welch
    frecuencias_welch, psd = welch(RR, fs=FS, window='hamming', nperseg=1200, noverlap=600, nfft=1200, detrend = 'linear')


    # Definición de bandas de frecuencia
    banda_vlf = (0.0033, 0.04)
    banda_lf = (0.04, 0.15)
    banda_hf = (0.15, 0.4)
    
    # Calcular potencia y frecuencia pico en cada banda, d
    potencia_vlf, pico_vlf = calcular_potencia_banda(frecuencias_welch, psd, banda_vlf)
    potencia_lf, pico_lf = calcular_potencia_banda(frecuencias_welch, psd, banda_lf)
    potencia_hf, pico_hf = calcular_potencia_banda(frecuencias_welch, psd, banda_hf)


    # Potencia total
    potencia_total = potencia_vlf + potencia_lf + potencia_hf
    
    # Calcular el porcentaje de cada banda
    porcentaje_vlf = (potencia_vlf / potencia_total) * 100
    porcentaje_lf = (potencia_lf / potencia_total) * 100
    porcentaje_hf = (potencia_hf / potencia_total) * 100
    
    # Potencia en unidades normalizadas
    potencia_lf_nu = (potencia_lf / (potencia_lf + potencia_hf)) * 100
    potencia_hf_nu = (potencia_hf / (potencia_lf + potencia_hf)) * 100
    
    # Cociente LF/HF
    lf_hf_ratio = potencia_lf / potencia_hf if potencia_hf != 0 else 0
    
    # Potencia en escala logarítmica
    potencia_vlf_log = np.log(potencia_vlf) if potencia_vlf > 0 else 0
    potencia_lf_log = np.log(potencia_lf) if potencia_lf > 0 else 0
    potencia_hf_log = np.log(potencia_hf) if potencia_hf > 0 else 0



    # Imprimir resultados
    print(f'VLF - Potencia: {potencia_vlf:.4f} ms², Frecuencia Pico: {pico_vlf:.4f} Hz, Potencia (log): {potencia_vlf_log:.4f}, Porcentaje: {porcentaje_vlf:.2f}%')
    print(f'LF - Potencia: {potencia_lf:.4f} ms², Frecuencia Pico: {pico_lf:.4f} Hz, Potencia (log): {potencia_lf_log:.4f}, Porcentaje: {porcentaje_lf:.2f}%')
    print(f'HF - Potencia: {potencia_hf:.4f} ms², Frecuencia Pico: {pico_hf:.4f} Hz, Potencia (log): {potencia_hf_log:.4f}, Porcentaje: {porcentaje_hf:.2f}%')
    print(f'Potencia total: {potencia_total:.4f} ms²')
    print(f'Potencia LF (nu): {potencia_lf_nu:.2f}')
    print(f'Potencia HF (nu): {potencia_hf_nu:.2f}')
    print(f'Ratio LF/HF: {lf_hf_ratio:.4f}')

    # Retornar los valores calculados si deseas usarlos en otros lugares
    return {
        "potencia_vlf": float(potencia_vlf),
        "pico_vlf": float(pico_vlf),
        "potencia_vlf_log": float(potencia_vlf_log),
        "porcentaje_vlf": float(porcentaje_vlf),
        "potencia_lf": float(potencia_lf),
        "pico_lf": float(pico_lf),
        "potencia_lf_log": float(potencia_lf_log),
        "porcentaje_lf": float(porcentaje_lf),
        "potencia_hf": float(potencia_hf),
        "pico_hf": float(pico_hf),
        "potencia_hf_log": float(potencia_hf_log),
        "porcentaje_hf": float(porcentaje_hf),
        "potencia_total": float(potencia_total),
        "potencia_lf_nu": float(potencia_lf_nu),
        "potencia_hf_nu": float(potencia_hf_nu),
        "lf_hf_ratio": float(lf_hf_ratio)
    }, frecuencias_welch, psd

import numpy as np

def reemplazar_nan_y_convertir(data):
    if isinstance(data, dict):
        return {k: reemplazar_nan_y_convertir(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [reemplazar_nan_y_convertir(item) for item in data]
    # Para enteros de numpy
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    # Para flotantes de numpy o float nativos
    elif isinstance(data, (np.float64, np.float32, float)):
        return None if math.isnan(data) else float(data)
    else:
        return data


# Vista para marcar un paciente como completado
@login_required
def complete_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, pk=paciente_id, user=request.user)
    if request.method == 'POST':
        paciente.fecha_nacimiento = timezone.now()
        paciente.save()
        return redirect('pacientes')


# Vista del perfil del especialista
@login_required
def perfil_doc(request):
    user = request.user
    try:
        doctor = Especialista.objects.get(user=user)
        context = {
            'nombre_especialista': doctor.nombre_especialista,
            'apellido_paterno': doctor.apellido_paterno,
            'apellido_materno': doctor.apellido_materno,
            'fecha_nacimiento': doctor.fecha_nacimiento,
            'departamento': doctor.departamento_id,  # Cambié a la forma correcta
            'username': user.username,
        }
        return render(request, 'perfil_especialista.html', context)
    except Especialista.DoesNotExist:
        return redirect('error_page')

def error_page(request):
    return render(request, 'error_page.html')

@login_required
def homeDoctor(request):
    especialista = Especialista.objects.get(user=request.user)  # Obtiene el especialista actual
    return render(request, 'homeDoctor.html', {'especialista': especialista})

@login_required
@csrf_exempt
def ver_grafico(request, ecg_id):
    banda = float(request.GET.get('banda', 300))  # Valor predeterminado de 10000 muestras
    banda_tacograma_segundos = float(request.GET.get('banda_tacograma', 300))  # Valor predeterminado para tacograma
    tiempo_requerido_ms  = banda_tacograma_segundos * 1000
    banda_analisis = int(request.GET.get('banda_analisis', 300)) #Análisis por default de 300 segundos
    try:
        registro_ecg = get_object_or_404(ECG, id_ecg = ecg_id)
        ecg_id = registro_ecg.id_ecg
        paciente = registro_ecg.paciente
        especialista = paciente.especialista
    except Exception as e:
        print(f"Error al obtener el registro ECG: {e}")
        return render(request, 'ver_grafico.html', {'mensaje': 'No se pudo encontrar el registro ECG.'})
    ecg_path = registro_ecg.archivo_ecg.path
    datos_ecg = formato_ecg(ecg_path)

    if datos_ecg is None:
        return render(request, 'ver_grafico.html', {'mensaje': 'El archivo ECG no pudo ser leído.'})
       # return render(request, 'ver_grafico.html', {'mensaje': 'El archivo ECG no tiene la estructura correcta.'})

    datos_ecg.iloc[:, 0] = pd.to_numeric(datos_ecg.iloc[:, 0], errors='coerce')
    datos_ecg.iloc[:, 1] = pd.to_numeric(datos_ecg.iloc[:, 1], errors='coerce')


    # 0.000	2.331
    tiempo_ECG = datos_ecg.iloc[:, 0]
    voltaje_ECG = datos_ecg.iloc[:, 1]

    fm = 1 /(tiempo_ECG.iloc[1]-tiempo_ECG.iloc[0])
    voltaje_F = filtro_pasa_bajas(voltaje_ECG, fm)
    voltaje, umbral = algoritmo_segunda_derivada(voltaje_F)
    picosRR, intervalosRR = detección_RR(voltaje_F, tiempo_ECG, fm)


   #print(f'Los intervalos RR totales son: {intervalosRR}')


    #lOS UNICOS RESULTADOS QUE NECESITAN UN RECALCULO...
    if request.method == 'POST':
        try:
            data = json.loads(request.body)#solicitud json
            inicio = int(data.get('inicio', 0))*1000
            fin = int(data.get('fin', 0))*1000
            #llegan correctamente
            indices_tramo = []
            tramo = 0
            print(f'Inicio: {inicio}, Fin: {fin}')
            for i, intervalo in enumerate(intervalosRR):
                tramo += intervalo
              #  print(f'Iteración {i}: Tramo acumulado: {tramo}')
                if inicio <= tramo <= fin:
                    indices_tramo.append(i)

            Tramo_intervalosRR = [float(intervalosRR[i]) for i in indices_tramo]
            Tramo_picosRR = [int(picosRR[i]) for i in indices_tramo]
            Tramo_picosRR = np.array(Tramo_picosRR)
            Tramo_intervalosRR = np.array(Tramo_intervalosRR)

            #Por el momento ya realiza la actualización del segmento enviado
            #Ahora solo es que se actualicen los nuevos datos ..

           #print(Tramo_picosRR)
            #El tramo numero 79 corresponde el intervalo que se limita en un inicio con 100 ms..
            #Y el tramo de tiempo ECG tiene otra linea de tiempo porque el tacograma y el ECG son tramos diferentes 
            #print(Tramo_intervalosRR)

            resultadosDT, SDNN = Parametro_dominio_tiempo(Tramo_intervalosRR)
          #  print(f'Los resultados en el dominio del tiempo : {resultadosDT}')
            fs, TR, RRuniform = Parametro_dominio_frecuencia(Tramo_picosRR, Tramo_intervalosRR, tiempo_ECG) #vamos a darle todo el segmento de tiepo y el filtrara los corresppondoetes a los picos 
            resultadosDF, frecuencias_welch, psd = calculo_welch(RRuniform, fs)
            HR = (60/Tramo_intervalosRR)*1000 # Frecuencia cardiaca
            print(f'psd: {psd}')
            psd = psd/100000
            rr_mean = np.mean(Tramo_intervalosRR)
            total_intervalos = len(Tramo_intervalosRR)
            print(resultadosDT)

            fig_histogramaRR = {
            'data': [
                {'x': Tramo_intervalosRR.tolist(), 'type': 'histogram'}
            ],
            'layout': {
                'title': 'Histograma - Intervalos RR',
                'xaxis': {'title': 'Intervalo RR (ms)'},
                'yaxis': {'title': 'Frecuencia'}
            }
        }
            
            fig_histogramaHR = {
            'data' : [
                {'x': HR.tolist(), 'type': 'histogram'}
            ],
            'layout': {
                'title': 'Histograma Frecuencia Cardiaca',
                'xaxis':{'title': 'FC (lpm)'},
                'yaxis': {'title': 'Frecuencia'}
            },
        }
            fig_welch = {
                'data': [
                    {'x': frecuencias_welch.tolist(), 'y': psd.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Welch ECG'}
                ],
                'layout': {
                    'title': 'Welch ',
                    'xaxis': {'title': 'Frecuencia (Hz)'},
                    'yaxis': {'title': 'Amplitud (ms2/Hz)'}
                }
            }
            respuesta = {
                'graph_json_Welch' : json.dumps(fig_welch),
                'graph_json_RR': json.dumps(fig_histogramaRR),
                'graph_json_HR': json.dumps(fig_histogramaHR),
                'resultadosDT': resultadosDT,
                'resultadosDF': resultadosDF,
                'frecuencias_welch': frecuencias_welch.tolist(),
                'psd': psd.tolist(),
                'HR': HR.tolist(),
                'rr_mean': rr_mean,
                'total_intervalos': total_intervalos,
                'mensaje': 'Cálculos realizados correctamente.'
            }

            respuesta = reemplazar_nan_y_convertir(respuesta)
            respuesta_json = json.dumps(respuesta)
         #   print("Respuesta JSON:", respuesta_json)
            return JsonResponse(respuesta)

        except Exception as e:
            print(f"Error al calcular parámetros: {e}")
            return render(request,'ver_grafico.html', {'mensaje': 'Error al calcular parámetros.'})
    else:
        Banda_Analisis = banda_analisis * 1000 #conversión a milisegundos
        Tiempo_ECG = tiempo_ECG[:Banda_Analisis]
        picos_acumulados = 0 
        duracion_intervalos = 0
        for intervalo in intervalosRR:
            duracion_intervalos += intervalo
            picos_acumulados += 1 #cada intervalo equivale a un punto R
            if duracion_intervalos >= Banda_Analisis:
                break
        picos_RR = picosRR[:picos_acumulados]
        intervalos_RR = intervalosRR[:picos_acumulados]
                # Calcular parámetros de dominio de tiempo y frecuencia

        resultadosDT, SDNN = Parametro_dominio_tiempo(intervalos_RR)
        print(resultadosDT)
        fs, TR, RRuniform = Parametro_dominio_frecuencia(picos_RR, intervalos_RR, Tiempo_ECG)
        resultadosDF, frecuencias_welch, psd = calculo_welch(RRuniform, fs)
        HR = (60/intervalosRR)*1000 # Frecuencia cardiaca
        psd = psd/100000
        rr_mean = np.mean(intervalos_RR)
        total_intervalos = len(intervalos_RR)
        print(f'SDNN: {SDNN}')
        duracion_acumulada = 0
        puntos_seleccionados = 0

        for intervalo in intervalosRR:
            duracion_acumulada += intervalo
            puntos_seleccionados += 1
            if duracion_acumulada >= tiempo_requerido_ms:
                break
        Tramo_picosRR = picosRR[:puntos_seleccionados]
        Tramo_intervalosRR = intervalosRR[:puntos_seleccionados]
       
        Tramo_picosRR = np.array(Tramo_picosRR)
        Tramo_intervalosRR = np.array(Tramo_intervalosRR)



    muestras = int(banda * fm)


    datos_completos = {
        "ecg": {
            "tiempo": tiempo_ECG.astype(float).tolist(),
            "voltaje": voltaje_ECG.astype(float).tolist(),
        },
        "tacograma": {
            "picos": picosRR.astype(float).tolist(),
            "intervalos": intervalosRR.astype(float).tolist(),
        },
        'fm': float(fm),  # Asegúrate de que 'fm' también sea un valor flotante
    }

    # Datos iniciales para gráficos
    tiempo_mostrar = tiempo_ECG[:muestras]
    voltaje_mostrar = voltaje_ECG[:muestras]

    fig_ecg = {
        'data': [
            {'x': tiempo_mostrar.tolist(), 'y': voltaje_mostrar.tolist(), 'type': 'scatter', 'mode': 'lines', 'name': 'ECG'}
        ],
        'layout': {
            'title': 'Gráfico ECG',
            'xaxis': {'title': 'Tiempo'},
            'yaxis': {'title': 'Voltaje'}
        }
    }

    fig_tacograma = {
        'data': [
            {'x': Tramo_picosRR.tolist(), 'y': Tramo_intervalosRR.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Tacograma'}
        ],
        'layout': {
            'title': 'Tacograma - Intervalos RR',
            'xaxis': {'title': 'Tiempo (s)'},
            'yaxis': {'title': 'Intervalo RR (ms)'}
        }
    }
    fig_histogramaRR = {
        'data': [
            {'x': Tramo_intervalosRR.tolist(), 'type': 'histogram'}
        ],
        'layout': {
            'title': 'Histograma - Intervalos RR',
            'xaxis': {'title': 'Intervalo RR (ms)'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }
    
    fig_histogramaHR = {
        'data' : [
            {'x': HR.tolist(), 'type': 'histogram'}
        ],
        'layout': {
            'title': 'Histograma Frecuencia Cardiaca',
            'xaxis':{'title': 'FC (lpm)'},
            'yaxis': {'title': 'Frecuencia'}
        },
    }
    fig_welch = {
        'data': [
            {'x': frecuencias_welch.tolist(), 'y': psd.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Welch ECG'}
        ],
        'layout': {
            'title': 'Welch ',
            'xaxis': {'title': 'Frecuencia (Hz)'},
            'yaxis': {'title': 'Amplitud (ms2/Hz)'}
        }
    }
    
    # Convertir el ndarray a lista
    #fig_tacograma_list = fig_tacograma.tolist()
    return render(request, 'ver_grafico.html', {
        'banda_analisis': banda_analisis,
        'fm': fm,
        'rr_mean' : rr_mean,
        'total_intervalos': total_intervalos,
        'paciente': paciente,
        'registro_ecg': registro_ecg,
        'datos_completos_json': json.dumps(datos_completos),
        'banda': banda,
        'graph_json': json.dumps(fig_ecg),
        'graph_json_tacograma': json.dumps(fig_tacograma),
        'banda_tacograma': banda_tacograma_segundos,
        'resultadosDT': (resultadosDT),
        'resultadosDF': (resultadosDF),
        'graph_json_RR': json.dumps(fig_histogramaRR),
        'graph_json_HR': json.dumps(fig_histogramaHR),
        'graph_json_Welch': json.dumps(fig_welch),
        'especialista': especialista,  # Datos del especialista
        'paciente': paciente,  # Datos del paciente
        'ecg_id' : ecg_id,
    })


def visualizacion_informe(request, paciente_id):
    # Obtener el paciente y el especialista relacionado
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)
    especialista = paciente.especialista  # Obtener el especialista relacionado con el paciente
    # Solo mostrar los datos del especialista y paciente
    return render(request, 'visualizacion_informe.html', {
        'especialista': especialista,  # Datos del especialista
        'paciente': paciente,  # Datos del paciente
    })

def generar_pdf(request):
    # Renderizar el contenido HTML del template
    html_content = render_to_string('ver_grafico.html', {'is_pdf': True})

    # Crear la respuesta HTTP para el PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="informe.pdf"'

    # Generar el PDF desde el contenido HTML
    HTML(string=html_content).write_pdf(response)

    return response
