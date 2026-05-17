package com.singvoicecolombia.datasetcapture.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max

data class DynamicRecognitionResult(
    val label: String,
    val confidence: Float,
    val margin: Float,
    val ready: Boolean,
    val collectedFrames: Int,
    val requiredFrames: Int,
    val handPresent: Boolean = true,
    val hasMotion: Boolean = true
)

class DynamicRecognitionService(private val context: Context) {
    private val labels: List<String> = loadLabels()
    private var interpreter: Interpreter? = loadInterpreterOrNull()
    private val frameBuffer = ArrayDeque<Array<Array<FloatArray>>>()
    private var lastMotionScore = Float.MAX_VALUE

    fun isModelReady(): Boolean = interpreter != null && labels.isNotEmpty()

    fun reset() {
        frameBuffer.clear()
        lastMotionScore = Float.MAX_VALUE
    }

    fun classify(imageProxy: ImageProxy, mirrorHorizontally: Boolean = false): DynamicRecognitionResult? {
        val currentInterpreter = interpreter
        if (currentInterpreter == null || labels.isEmpty()) {
            imageProxy.close()
            return null
        }

        return try {
            val inputShape = currentInterpreter.getInputTensor(0).shape()
            val outputShape = currentInterpreter.getOutputTensor(0).shape()
            val inputHeight = inputShape.getOrNull(1) ?: 96
            val inputWidth = inputShape.getOrNull(2) ?: 96
            val inputChannels = inputShape.getOrNull(3) ?: 32
            val requiredFrames = requiredFramesFor(inputChannels)
            val outputSize = outputShape.lastOrNull() ?: labels.size

            val frameBitmap = imageProxyToBitmap(imageProxy, mirrorHorizontally)
            if (!hasHandInCenterRoi(frameBitmap)) {
                frameBitmap.recycle()
                frameBuffer.clear()
                return DynamicRecognitionResult(
                    label = "",
                    confidence = 0f,
                    margin = 0f,
                    ready = true,
                    collectedFrames = 0,
                    requiredFrames = requiredFrames,
                    handPresent = false,
                    hasMotion = false
                )
            }

            val frameFeatures = buildFrameFeaturesFromBitmap(frameBitmap, inputHeight, inputWidth)
            frameBitmap.recycle()
            frameBuffer.addLast(frameFeatures)

            while (frameBuffer.size > requiredFrames) {
                frameBuffer.removeFirst()
            }

            if (frameBuffer.size < requiredFrames) {
                DynamicRecognitionResult(
                    label = "",
                    confidence = 0f,
                    margin = 0f,
                    ready = false,
                    collectedFrames = frameBuffer.size,
                    requiredFrames = requiredFrames
                )
            } else {
                val input = buildInputFromBuffer(inputHeight, inputWidth, inputChannels)
                if (inputChannels == MOTION_SUMMARY_CHANNELS && lastMotionScore < MIN_MOTION_SCORE) {
                    return DynamicRecognitionResult(
                        label = "",
                        confidence = 0f,
                        margin = 0f,
                        ready = true,
                        collectedFrames = frameBuffer.size,
                        requiredFrames = requiredFrames,
                        hasMotion = false
                    )
                }

                val output = Array(1) { FloatArray(outputSize) }
                currentInterpreter.run(input, output)
                val probabilities = normalizeOutput(output[0])
                val rankedIndexes = probabilities.indices.sortedByDescending { probabilities[it] }
                val bestIndex = rankedIndexes.firstOrNull() ?: 0
                val secondIndex = rankedIndexes.getOrNull(1)
                val label = labels.getOrElse(bestIndex) { "?" }
                val confidence = probabilities[bestIndex].coerceIn(0f, 1f)
                val secondConfidence = secondIndex?.let { probabilities[it].coerceIn(0f, 1f) } ?: 0f

                DynamicRecognitionResult(
                    label = label,
                    confidence = confidence,
                    margin = (confidence - secondConfidence).coerceAtLeast(0f),
                    ready = true,
                    collectedFrames = frameBuffer.size,
                    requiredFrames = requiredFrames
                )
            }
        } catch (error: Exception) {
            Log.e(TAG, "Error durante reconocimiento dinamico.", error)
            null
        } finally {
            imageProxy.close()
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        frameBuffer.clear()
    }

    private fun loadInterpreterOrNull(): Interpreter? {
        return try {
            Interpreter(loadModelBuffer(MODEL_FILE_NAME)).also {
                Log.i(TAG, "Modelo dinamico cargado desde assets.")
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

    private fun buildFrameFeaturesFromBitmap(
        bitmap: Bitmap,
        targetHeight: Int,
        targetWidth: Int
    ): Array<Array<FloatArray>> {
        val lumaGrid = resizeLumaGrid(cropCenterLumaGrid(bitmap), targetWidth, targetHeight)
        val frameFeatures = Array(targetHeight) {
            Array(targetWidth) {
                FloatArray(CHANNELS_PER_FRAME)
            }
        }

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                frameFeatures[y][x][0] = lumaGrid[y][x]
                frameFeatures[y][x][1] = edgeMagnitude(lumaGrid, x, y)
            }
        }

        return frameFeatures
    }

    private fun hasHandInCenterRoi(bitmap: Bitmap): Boolean {
        val roiSize = (minOf(bitmap.width, bitmap.height) * ROI_RATIO).toInt().coerceAtLeast(1)
        val roiLeft = (bitmap.width - roiSize) / 2
        val roiTop = (bitmap.height - roiSize) / 2
        val step = max(1, roiSize / HAND_SKIN_SAMPLE_SIZE)
        var skinPixels = 0
        var sampledPixels = 0

        for (y in roiTop until roiTop + roiSize step step) {
            for (x in roiLeft until roiLeft + roiSize step step) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                val luma = 0.299f * r + 0.587f * g + 0.114f * b
                val cb = 128f + (b - luma) * 0.564f
                val cr = 128f + (r - luma) * 0.713f

                if (cr in 133f..173f && cb in 77f..127f) {
                    skinPixels += 1
                }
                sampledPixels += 1
            }
        }

        if (sampledPixels == 0) return false
        val skinRatio = skinPixels.toFloat() / sampledPixels.toFloat()
        return skinRatio >= HAND_MIN_AREA_RATIO
    }

    private fun cropCenterLumaGrid(bitmap: Bitmap): Array<FloatArray> {
        val roiSize = (minOf(bitmap.width, bitmap.height) * ROI_RATIO).toInt().coerceAtLeast(1)
        val roiLeft = (bitmap.width - roiSize) / 2
        val roiTop = (bitmap.height - roiSize) / 2
        val lumaGrid = Array(roiSize) { FloatArray(roiSize) }

        for (y in 0 until roiSize) {
            for (x in 0 until roiSize) {
                val pixel = bitmap.getPixel(roiLeft + x, roiTop + y)
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
            buffer.get(rowBytes, 0, minOf(rowBytes.size, buffer.remaining()))
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

    private fun readPlaneValue(plane: ImageProxy.PlaneProxy, x: Int, y: Int): Int {
        val index = y * plane.rowStride + x * plane.pixelStride
        val safeIndex = index.coerceIn(0, plane.buffer.limit() - 1)
        return plane.buffer.get(safeIndex).toInt() and 0xFF
    }

    private fun buildInputFromBuffer(
        targetHeight: Int,
        targetWidth: Int,
        inputChannels: Int
    ): Array<Array<Array<FloatArray>>> {
        val input = Array(1) {
            Array(targetHeight) {
                Array(targetWidth) {
                    FloatArray(inputChannels)
                }
            }
        }
        lastMotionScore = Float.MAX_VALUE

        if (inputChannels == MOTION_SUMMARY_CHANNELS) {
            fillMotionSummaryInput(input, targetHeight, targetWidth)
            return input
        }

        frameBuffer.forEachIndexed { frameIndex, frameFeatures ->
            val channelOffset = frameIndex * CHANNELS_PER_FRAME
            for (y in 0 until targetHeight) {
                for (x in 0 until targetWidth) {
                    if (channelOffset < inputChannels) {
                        input[0][y][x][channelOffset] = frameFeatures[y][x][0]
                    }
                    if (channelOffset + 1 < inputChannels) {
                        input[0][y][x][channelOffset + 1] = frameFeatures[y][x][1]
                    }
                }
            }
        }

        return input
    }

    private fun fillMotionSummaryInput(
        input: Array<Array<Array<FloatArray>>>,
        targetHeight: Int,
        targetWidth: Int
    ) {
        val frames = frameBuffer.toList()
        val firstFrame = frames.first()
        val lastFrame = frames.last()
        val motionGrid = Array(targetHeight) { FloatArray(targetWidth) }
        val divisor = (frames.size - 1).coerceAtLeast(1).toFloat()
        var motionTotal = 0f

        for (index in 0 until frames.lastIndex) {
            val currentFrame = frames[index]
            val nextFrame = frames[index + 1]
            for (y in 0 until targetHeight) {
                for (x in 0 until targetWidth) {
                    motionGrid[y][x] += abs(nextFrame[y][x][0] - currentFrame[y][x][0])
                }
            }
        }

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val motionValue = (motionGrid[y][x] / divisor).coerceIn(0f, 255f)
                motionGrid[y][x] = motionValue
                motionTotal += motionValue

                input[0][y][x][0] = firstFrame[y][x][0]
                input[0][y][x][1] = firstFrame[y][x][1]
                input[0][y][x][2] = lastFrame[y][x][0]
                input[0][y][x][3] = lastFrame[y][x][1]
                input[0][y][x][4] = motionValue
            }
        }

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                input[0][y][x][5] = edgeMagnitude(motionGrid, x, y)
            }
        }

        lastMotionScore = motionTotal / (targetHeight * targetWidth).coerceAtLeast(1).toFloat()
    }

    private fun edgeMagnitude(lumaGrid: Array<FloatArray>, x: Int, y: Int): Float {
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

    private fun requiredFramesFor(inputChannels: Int): Int {
        return if (inputChannels == MOTION_SUMMARY_CHANNELS) {
            MOTION_SEQUENCE_LENGTH
        } else {
            (inputChannels / CHANNELS_PER_FRAME).coerceAtLeast(1)
        }
    }

    companion object {
        private const val TAG = "DynamicRecognition"
        private const val MODEL_FILE_NAME = "modelo_dinamico.tflite"
        private const val LABELS_FILE_NAME = "dynamic_labels.txt"
        private const val CHANNELS_PER_FRAME = 2
        private const val MOTION_SUMMARY_CHANNELS = 6
        private const val MOTION_SEQUENCE_LENGTH = 16
        private const val MIN_MOTION_SCORE = 2.0f
        private const val ROI_RATIO = 0.75f
        private const val HAND_MIN_AREA_RATIO = 0.015f
        private const val HAND_SKIN_SAMPLE_SIZE = 160
    }
}
