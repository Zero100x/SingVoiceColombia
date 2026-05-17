package com.singvoicecolombia.datasetcapture.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

data class AlphabetRecognitionResult(
    val letter: String,
    val confidence: Float,
    val margin: Float,
    val signPresent: Boolean
)

class AlphabetRecognitionService(private val context: Context) {
    private val labels: List<String> = loadLabels()
    private var interpreter: Interpreter? = loadInterpreterOrNull()
    private var handLandmarker: HandLandmarker? = loadHandLandmarkerOrNull()
    private var noHandLogCounter = 0
    private var predictionLogCounter = 0

    fun isModelReady(): Boolean = isAlphabetModelReady() && isHandDetectorReady()

    fun isAlphabetModelReady(): Boolean = interpreter != null && labels.isNotEmpty()

    fun isHandDetectorReady(): Boolean = handLandmarker != null

    fun classify(imageProxy: ImageProxy, mirrorHorizontally: Boolean = false): AlphabetRecognitionResult? {
        val currentInterpreter = interpreter
        val currentHandLandmarker = handLandmarker
        if (currentInterpreter == null || labels.isEmpty() || currentHandLandmarker == null) {
            imageProxy.close()
            return null
        }

        return try {
            val inputShape = currentInterpreter.getInputTensor(0).shape()
            val outputShape = currentInterpreter.getOutputTensor(0).shape()
            val inputHeight = inputShape.getOrNull(1) ?: 224
            val inputWidth = inputShape.getOrNull(2) ?: 224
            val inputChannels = inputShape.getOrNull(3) ?: 3
            val outputSize = outputShape.lastOrNull() ?: labels.size

            val frameBitmap = imageProxyToBitmap(imageProxy, mirrorHorizontally)
            val handBitmap = cropHandWithMediaPipe(frameBitmap, currentHandLandmarker)
            frameBitmap.recycle()

            if (handBitmap == null) {
                logNoHandDetected()
                return AlphabetRecognitionResult(
                    letter = "",
                    confidence = 0f,
                    margin = 0f,
                    signPresent = false
                )
            }

            val output = Array(1) { FloatArray(outputSize) }
            val modelInput = buildInputFromBitmap(handBitmap, inputHeight, inputWidth, inputChannels)
            handBitmap.recycle()
            currentInterpreter.run(modelInput, output)

            val probabilities = normalizeOutput(output[0])
            val rankedIndexes = probabilities.indices.sortedByDescending { probabilities[it] }
            val bestIndex = chooseBestIndex(rankedIndexes, probabilities)
            val secondIndex = rankedIndexes.firstOrNull { it != bestIndex }
            val label = labels.getOrElse(bestIndex) { "?" }
            val confidence = probabilities[bestIndex].coerceIn(0f, 1f)
            val secondConfidence = secondIndex?.let { probabilities[it].coerceIn(0f, 1f) } ?: 0f
            val margin = (confidence - secondConfidence).coerceAtLeast(0f)

            logPrediction(label, confidence, margin, rankedIndexes, probabilities)

            AlphabetRecognitionResult(
                letter = label,
                confidence = confidence,
                margin = margin,
                signPresent = true
            )
        } catch (error: Exception) {
            Log.e(TAG, "Error durante reconocimiento de alfabeto.", error)
            null
        } finally {
            imageProxy.close()
        }
    }

    private fun logNoHandDetected() {
        noHandLogCounter += 1
        if (noHandLogCounter % NO_HAND_LOG_INTERVAL == 0) {
            Log.d(TAG, "MediaPipe no detecto mano en el frame.")
        }
    }

    private fun logPrediction(
        label: String,
        confidence: Float,
        margin: Float,
        rankedIndexes: List<Int>,
        probabilities: FloatArray
    ) {
        predictionLogCounter += 1
        if (predictionLogCounter % PREDICTION_LOG_INTERVAL != 0) return

        val topPredictions = rankedIndexes.take(3).joinToString(separator = ", ") { index ->
            val topLabel = labels.getOrElse(index) { "?" }
            val percent = (probabilities[index].coerceIn(0f, 1f) * 100f).toInt()
            "$topLabel=$percent%"
        }
        Log.d(
            TAG,
            "Prediccion cruda: clase=$label, confianza=${confidence}, margen=${margin}, top3=[$topPredictions]"
        )
    }

    private fun chooseBestIndex(rankedIndexes: List<Int>, probabilities: FloatArray): Int {
        val bestIndex = rankedIndexes.firstOrNull() ?: 0
        val bestLabel = labels.getOrElse(bestIndex) { "" }.trim().uppercase()
        if (bestLabel != "NN") return bestIndex

        val secondIndex = rankedIndexes.firstOrNull { labels.getOrElse(it) { "" }.trim().uppercase() != "NN" }
            ?: return bestIndex
        val nnConfidence = probabilities[bestIndex].coerceIn(0f, 1f)
        val secondConfidence = probabilities[secondIndex].coerceIn(0f, 1f)
        val nnMargin = nnConfidence - secondConfidence

        return if (nnConfidence >= MIN_NN_CONFIDENCE && nnMargin >= MIN_NN_MARGIN) {
            bestIndex
        } else {
            secondIndex
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        handLandmarker?.close()
        handLandmarker = null
    }

    private fun loadInterpreterOrNull(): Interpreter? {
        return try {
            Interpreter(loadModelBuffer(MODEL_FILE_NAME)).also {
                Log.i(TAG, "Modelo de alfabeto cargado desde assets.")
            }
        } catch (error: Exception) {
            Log.e(TAG, "No se pudo cargar $MODEL_FILE_NAME desde assets.", error)
            null
        }
    }

    private fun loadModelBuffer(fileName: String): ByteBuffer {
        return try {
            loadMappedModelFile(fileName)
        } catch (error: Exception) {
            Log.w(TAG, "No se pudo mapear $fileName; se cargara como bytes.", error)
            loadModelBytes(fileName)
        }
    }

    private fun loadMappedModelFile(fileName: String): MappedByteBuffer {
        context.assets.openFd(fileName).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).use { input ->
                return input.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength
                )
            }
        }
    }

    private fun loadModelBytes(fileName: String): ByteBuffer {
        val bytes = context.assets.open(fileName).use { input ->
            input.readBytes()
        }
        return ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .apply {
                put(bytes)
                rewind()
            }
    }

    private fun loadLabels(): List<String> {
        return try {
            context.assets.open(LABELS_FILE_NAME).bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotBlank() }
                    .toList()
            }
        } catch (error: Exception) {
            Log.e(TAG, "No se pudieron cargar las etiquetas $LABELS_FILE_NAME.", error)
            emptyList()
        }
    }

    private fun loadHandLandmarkerOrNull(): HandLandmarker? {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(HAND_LANDMARKER_TASK)
                .build()
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(MIN_HAND_DETECTION_CONFIDENCE)
                .setMinHandPresenceConfidence(MIN_HAND_PRESENCE_CONFIDENCE)
                .setMinTrackingConfidence(MIN_HAND_TRACKING_CONFIDENCE)
                .build()

            HandLandmarker.createFromOptions(context, options).also {
                Log.i(TAG, "MediaPipe HandLandmarker cargado desde assets.")
            }
        } catch (error: Exception) {
            Log.e(TAG, "No se pudo cargar $HAND_LANDMARKER_TASK.", error)
            null
        }
    }

    private fun cropHandWithMediaPipe(
        frameBitmap: Bitmap,
        currentHandLandmarker: HandLandmarker
    ): Bitmap? {
        val mpImage = BitmapImageBuilder(frameBitmap).build()
        val result = currentHandLandmarker.detect(mpImage)
        val landmarks = result.landmarks().firstOrNull() ?: return null

        var minX = 1f
        var minY = 1f
        var maxX = 0f
        var maxY = 0f

        for (landmark in landmarks) {
            minX = min(minX, landmark.x())
            minY = min(minY, landmark.y())
            maxX = max(maxX, landmark.x())
            maxY = max(maxY, landmark.y())
        }

        if (maxX <= minX || maxY <= minY) return null

        val imageWidth = frameBitmap.width.toFloat()
        val imageHeight = frameBitmap.height.toFloat()
        val boxWidth = (maxX - minX) * imageWidth
        val boxHeight = (maxY - minY) * imageHeight

        if (!isPlausibleHandBox(boxWidth, boxHeight, imageWidth, imageHeight)) {
            Log.d(TAG, "MediaPipe detecto una caja no plausible para mano: ${boxWidth}x${boxHeight}.")
            return null
        }

        val padding = max(boxWidth, boxHeight) * HAND_BOX_PADDING_RATIO
        val centerX = ((minX + maxX) * 0.5f) * imageWidth
        val centerY = ((minY + maxY) * 0.5f) * imageHeight
        val side = max(boxWidth, boxHeight) + padding * 2f

        val left = (centerX - side * 0.5f).toInt().coerceIn(0, frameBitmap.width - 1)
        val top = (centerY - side * 0.5f).toInt().coerceIn(0, frameBitmap.height - 1)
        val right = (centerX + side * 0.5f).toInt().coerceIn(left + 1, frameBitmap.width)
        val bottom = (centerY + side * 0.5f).toInt().coerceIn(top + 1, frameBitmap.height)

        return Bitmap.createBitmap(frameBitmap, left, top, right - left, bottom - top)
    }

    private fun isPlausibleHandBox(
        boxWidth: Float,
        boxHeight: Float,
        imageWidth: Float,
        imageHeight: Float
    ): Boolean {
        val minImageSide = min(imageWidth, imageHeight)
        val longestBoxSide = max(boxWidth, boxHeight)
        val boxAreaRatio = (boxWidth * boxHeight) / (imageWidth * imageHeight).coerceAtLeast(1f)
        val tooSmall = longestBoxSide < minImageSide * MIN_HAND_BOX_SIDE_RATIO ||
            boxAreaRatio < MIN_HAND_BOX_AREA_RATIO

        return !tooSmall
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy, mirrorHorizontally: Boolean): Bitmap {
        val bitmap = when (imageProxy.format) {
            PixelFormat.RGBA_8888 -> imageProxyRgbaToBitmap(imageProxy)
            ImageFormat.YUV_420_888 -> imageProxyYuvToBitmap(imageProxy)
            else -> imageProxyYuvToBitmap(imageProxy)
        }
        return orientBitmap(bitmap, imageProxy.imageInfo.rotationDegrees, mirrorHorizontally)
    }

    private fun imageProxyRgbaToBitmap(imageProxy: ImageProxy): Bitmap {
        val plane = imageProxy.planes[0]
        val width = imageProxy.width
        val height = imageProxy.height
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val buffer = plane.buffer
        buffer.rewind()

        if (plane.rowStride == width * 4) {
            bitmap.copyPixelsFromBuffer(buffer)
            return bitmap
        }

        val rowBytes = ByteArray(plane.rowStride)
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            buffer.position(y * plane.rowStride)
            buffer.get(rowBytes, 0, min(rowBytes.size, buffer.remaining()))
            for (x in 0 until width) {
                val offset = x * plane.pixelStride
                val r = rowBytes[offset].toInt() and 0xFF
                val g = rowBytes[offset + 1].toInt() and 0xFF
                val b = rowBytes[offset + 2].toInt() and 0xFF
                val a = rowBytes[offset + 3].toInt() and 0xFF
                pixels[y * width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun imageProxyYuvToBitmap(imageProxy: ImageProxy): Bitmap {
        val width = imageProxy.width
        val height = imageProxy.height
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val yValue = readPlaneValue(yPlane, x, y)
                val chromaX = x / 2
                val chromaY = y / 2
                val uValue = readPlaneValue(uPlane, chromaX, chromaY) - 128
                val vValue = readPlaneValue(vPlane, chromaX, chromaY) - 128

                val r = (yValue + 1.402f * vValue).toInt().coerceIn(0, 255)
                val g = (yValue - 0.344136f * uValue - 0.714136f * vValue).toInt().coerceIn(0, 255)
                val b = (yValue + 1.772f * uValue).toInt().coerceIn(0, 255)
                pixels[y * width + x] = -0x1000000 or (r shl 16) or (g shl 8) or b
            }
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun orientBitmap(bitmap: Bitmap, rotationDegrees: Int, mirrorHorizontally: Boolean): Bitmap {
        val rotatedBitmap = rotateBitmap(bitmap, rotationDegrees)
        if (!mirrorHorizontally) return rotatedBitmap

        val matrix = Matrix().apply {
            preScale(-1f, 1f)
        }
        val mirrored = Bitmap.createBitmap(
            rotatedBitmap,
            0,
            0,
            rotatedBitmap.width,
            rotatedBitmap.height,
            matrix,
            true
        )
        if (mirrored != rotatedBitmap) {
            rotatedBitmap.recycle()
        }
        return mirrored
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap

        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        return rotated
    }

    private fun buildInputFromBitmap(
        bitmap: Bitmap,
        targetHeight: Int,
        targetWidth: Int,
        channels: Int
    ): Array<Array<Array<FloatArray>>> {
        val safeChannels = channels.coerceAtLeast(1)
        val input = Array(1) {
            Array(targetHeight) {
                Array(targetWidth) {
                    FloatArray(safeChannels)
                }
            }
        }
        val lumaGrid = resizeLumaGrid(bitmapToLumaGrid(bitmap), targetWidth, targetHeight)

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val luma = lumaGrid[y][x]
                input[0][y][x][0] = luma

                if (safeChannels > 1) {
                    input[0][y][x][1] = sobelEdgeMagnitude(lumaGrid, x, y)
                }

                for (channel in 2 until safeChannels) {
                    input[0][y][x][channel] = luma
                }
            }
        }

        return input
    }

    private fun bitmapToLumaGrid(bitmap: Bitmap): Array<FloatArray> {
        val lumaGrid = Array(bitmap.height) { FloatArray(bitmap.width) }
        for (y in 0 until bitmap.height) {
            for (x in 0 until bitmap.width) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                lumaGrid[y][x] = 0.299f * r + 0.587f * g + 0.114f * b
            }
        }
        return lumaGrid
    }

    private fun resizeLumaGrid(
        source: Array<FloatArray>,
        targetWidth: Int,
        targetHeight: Int
    ): Array<FloatArray> {
        val sourceHeight = source.size
        val sourceWidth = source.firstOrNull()?.size ?: 0
        if (sourceHeight == targetHeight && sourceWidth == targetWidth) return source
        if (sourceHeight == 0 || sourceWidth == 0) return Array(targetHeight) { FloatArray(targetWidth) }

        val resized = Array(targetHeight) { FloatArray(targetWidth) }
        val xScale = sourceWidth.toFloat() / targetWidth.toFloat()
        val yScale = sourceHeight.toFloat() / targetHeight.toFloat()

        for (targetY in 0 until targetHeight) {
            val startY = (targetY * yScale).toInt().coerceIn(0, sourceHeight - 1)
            val endY = ceil(((targetY + 1) * yScale).toDouble())
                .toInt()
                .coerceIn(startY + 1, sourceHeight)

            for (targetX in 0 until targetWidth) {
                val startX = (targetX * xScale).toInt().coerceIn(0, sourceWidth - 1)
                val endX = ceil(((targetX + 1) * xScale).toDouble())
                    .toInt()
                    .coerceIn(startX + 1, sourceWidth)

                var total = 0f
                var count = 0
                for (sourceY in startY until endY) {
                    for (sourceX in startX until endX) {
                        total += source[sourceY][sourceX]
                        count += 1
                    }
                }
                resized[targetY][targetX] = total / count.coerceAtLeast(1).toFloat()
            }
        }
        return resized
    }

    private fun readPlaneValue(plane: ImageProxy.PlaneProxy, x: Int, y: Int): Int {
        val index = y * plane.rowStride + x * plane.pixelStride
        val safeIndex = index.coerceIn(0, plane.buffer.limit() - 1)
        return plane.buffer.get(safeIndex).toInt() and 0xFF
    }

    private fun sobelEdgeMagnitude(lumaGrid: Array<FloatArray>, x: Int, y: Int): Float {
        val maxY = lumaGrid.lastIndex
        val maxX = lumaGrid.firstOrNull()?.lastIndex ?: 0
        val x0 = (x - 1).coerceAtLeast(0)
        val x1 = x
        val x2 = (x + 1).coerceAtMost(maxX)
        val y0 = (y - 1).coerceAtLeast(0)
        val y1 = y
        val y2 = (y + 1).coerceAtMost(maxY)

        val gx =
            -lumaGrid[y0][x0] + lumaGrid[y0][x2] +
            -2f * lumaGrid[y1][x0] + 2f * lumaGrid[y1][x2] +
            -lumaGrid[y2][x0] + lumaGrid[y2][x2]
        val gy =
            -lumaGrid[y0][x0] - 2f * lumaGrid[y0][x1] - lumaGrid[y0][x2] +
            lumaGrid[y2][x0] + 2f * lumaGrid[y2][x1] + lumaGrid[y2][x2]

        return ((abs(gx) + abs(gy)) * 0.5f).coerceIn(0f, 255f)
    }

    private fun normalizeOutput(rawValues: FloatArray): FloatArray {
        val alreadyProbabilities = rawValues.all { it in 0f..1f } && rawValues.sum() in 0.95f..1.05f
        if (alreadyProbabilities) return rawValues

        val maxValue = rawValues.maxOrNull() ?: 0f
        val exps = rawValues.map { exp((it - maxValue).toDouble()).toFloat() }
        val sum = exps.sum().takeIf { it > 0f } ?: 1f
        return exps.map { it / sum }.toFloatArray()
    }

    companion object {
        private const val TAG = "AlphabetRecognition"
        private const val MODEL_FILE_NAME = "modelo_alfabeto.tflite"
        private const val LABELS_FILE_NAME = "labels.txt"
        private const val HAND_LANDMARKER_TASK = "hand_landmarker.task"
        private const val MIN_HAND_DETECTION_CONFIDENCE = 0.5f
        private const val MIN_HAND_PRESENCE_CONFIDENCE = 0.5f
        private const val MIN_HAND_TRACKING_CONFIDENCE = 0.5f
        private const val HAND_BOX_PADDING_RATIO = 0.65f
        private const val MIN_HAND_BOX_SIDE_RATIO = 0.06f
        private const val MIN_HAND_BOX_AREA_RATIO = 0.0025f
        private const val PREDICTION_LOG_INTERVAL = 15
        private const val MIN_NN_CONFIDENCE = 0.85f
        private const val MIN_NN_MARGIN = 0.18f
        private const val NO_HAND_LOG_INTERVAL = 20
    }
}
