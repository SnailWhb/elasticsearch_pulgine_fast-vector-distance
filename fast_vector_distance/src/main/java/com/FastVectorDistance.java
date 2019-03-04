/**
 * \* Created with IntelliJ IDEA.
 * \* User: whb
 * \* Date: 19-3-2
 * \* Time: 下午6:06
 * \* Description:
 * \
 */
package com;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.LeafReaderContext;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.ScriptPlugin;
import org.elasticsearch.script.ScriptContext;
import org.elasticsearch.script.ScriptEngine;
import org.elasticsearch.script.SearchScript;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.store.ByteArrayDataInput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.*;


/**
 * \* Created with IntelliJ IDEA.
 * \* User: 王火斌
 * \* Date: 18-8-９
 * \* Time: 下午2:32
 * \* Description:为了得到个性化推荐搜索效果，我们计算用户向量与每个产品特征向量的相似度。
 * 　　　　　　　　　相似度越高，最后得到的分值越高，排序越靠前.
 * \
 */

/**
 * This class is instantiated when Elasticsearch loads the plugin for the
 * first time. If you change the name of this plugin, make sure to update
 * src/main/resources/es-plugin.properties file that points to this class.
 */
public final class FastVectorDistance extends Plugin implements ScriptPlugin {

    @Override
    public ScriptEngine getScriptEngine(Settings settings, Collection<ScriptContext<?>> contexts) {
        return new FastVectorDistanceEngine();
    }

    private static class FastVectorDistanceEngine implements ScriptEngine {
        private final static Logger logger = LogManager.getLogger(FastVectorDistance.class);
        private static final int DOUBLE_SIZE = 8;

        double queryVectorNorm;

        @Override
        public String getType() {
            return "feature_vector_scoring_script";
        }

        @Override
        public <T> T compile(String scriptName, String scriptSource, ScriptContext<T> context, Map<String, String> params) {
            logger.info("The feature_vector_scoring_script is calculating the similarity of users and commodities");
            if (!context.equals(SearchScript.CONTEXT)) {
                throw new IllegalArgumentException(getType() + " scripts cannot be used for context [" + context.name + "]");
            }
            if ("whb_fvd".equals(scriptSource)) {
                SearchScript.Factory factory = (p, lookup) -> new SearchScript.LeafFactory() {
                    // The field to compare against
                    final String field;
                    //Whether this search should be cosine or dot product
                    final Boolean cosine;
                    //The query embedded vector
                    final Object vector;
                    Boolean exclude;
                    //The final comma delimited vector representation of the query vector
                    double[] inputVector;

                    {
                        if (p.containsKey("field") == false) {
                            throw new IllegalArgumentException("Missing parameter [field]");
                        }

                        //Determine if cosine
                        final Object cosineBool = p.get("cosine");
                        cosine = cosineBool != null ? (boolean) cosineBool : true;

                        //Get the field value from the query
                        field = p.get("field").toString();

                        final Object excludeBool = p.get("exclude");
                        exclude = excludeBool != null ? (boolean) cosineBool : true;

                        //Get the query vector embedding
                        vector = p.get("vector");

                        //Determine if raw comma-delimited vector or embedding was passed
                        if (vector != null) {
                            final ArrayList<Double> tmp = (ArrayList<Double>) vector;
                            inputVector = new double[tmp.size()];
                            for (int i = 0; i < inputVector.length; i++) {
                                inputVector[i] = tmp.get(i);
                            }
                        } else {
                            final Object encodedVector = p.get("encoded_vector");
                            if (encodedVector == null) {
                                throw new IllegalArgumentException("Must have 'vector' or 'encoded_vector' as a parameter");
                            }
                            inputVector = Util.convertBase64ToArray((String) encodedVector);
                        }

                        //If cosine calculate the query vec norm
                        if (cosine) {
                            queryVectorNorm = 0d;
                            // compute query inputVector norm once
                            for (double v : inputVector) {
                                queryVectorNorm += Math.pow(v, 2.0);
                            }
                        }
                    }

                    @Override
                    public SearchScript newInstance(LeafReaderContext context) throws IOException {

                        return new SearchScript(p, lookup, context) {
                            Boolean is_value = false;

                            // Use Lucene LeafReadContext to access binary values directly.
                            BinaryDocValues accessor = context.reader().getBinaryDocValues(field);

                            @Override
                            public void setDocument(int docId) {
                                // advance has undefined behavior calling with a docid <= its current docid
                                try {
                                    accessor.advanceExact(docId);
                                    is_value = true;
                                } catch (IOException e) {
                                    is_value = false;
                                }
                            }


                            @Override
                            public double runAsDouble() {

                                //If there is no field value return 0 rather than fail.
                                if (!is_value) return 0.0d;

                                final int inputVectorSize = inputVector.length;
                                final byte[] bytes;

                                try {
                                    bytes = accessor.binaryValue().bytes;
                                } catch (IOException e) {
                                    return 0d;
                                }


                                final ByteArrayDataInput byteDocVector = new ByteArrayDataInput(bytes);

                                byteDocVector.readVInt();

                                final int docVectorLength = byteDocVector.readVInt(); // returns the number of bytes to read

                                if (docVectorLength != inputVectorSize * DOUBLE_SIZE) {
                                    return 0d;
                                }

                                final int position = byteDocVector.getPosition();

                                final DoubleBuffer doubleBuffer = ByteBuffer.wrap(bytes, position, docVectorLength).asDoubleBuffer();

                                final double[] docVector = new double[inputVectorSize];

                                doubleBuffer.get(docVector);

                                double docVectorNorm = 0d;
                                double score = 0d;

                                //calculate dot product of document vector and query vector
                                for (int i = 0; i < inputVectorSize; i++) {

                                    score += docVector[i] * inputVector[i];

                                    if (cosine) {
                                        docVectorNorm += Math.pow(docVector[i], 2.0);
                                    }
                                }

                                //If cosine, calcluate cosine score
                                if (cosine) {

                                    if (docVectorNorm == 0 || queryVectorNorm == 0) return 0d;

                                    score = score / (Math.sqrt(docVectorNorm) * Math.sqrt(queryVectorNorm));
                                }

                                return score;
                            }
                        };
                    }

                    @Override
                    public boolean needs_score() {
                        return false;
                    }
                };
                return context.factoryClazz.cast(factory);
            }
            throw new IllegalArgumentException("Unknown script name " + scriptSource);
        }

        @Override
        public void close() {}
    }
}
