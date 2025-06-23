// Unlearning Module
const UnlearningManager = (function() {
    const UNLEARNING_METHODS = {
        NO_UNLEARNING: 'No Unlearning',
        PARAMETER_RESETTING: 'Parameter Resetting',
        GRADIENT_ASCENT: 'Gradient Ascent',
    };
    const RETRAINING_METHODS = {
        NO_RETRAINING: 'No Retraining',
        KNOWLEDGE_DISTILLATION: 'Knowledge Distillation',
    };

    function updateUnlearnUI() {
        const unlearningFields = document.getElementById("unlearning-fields");
        const method = document.getElementById("unlearning-method-select").value;

        if (method === "No Unlearning") {
            unlearningFields.style.display = "none";
            return
        }
        else {
            unlearningFields.style.display = "block";
        }

        const elements = {
            unlearnAttackers: {
                title: document.getElementById("unlearn-attackers-title"),
                container: document.getElementById("unlearn-attackers-container"),
            },
            unlearningParticipants: {
                title: document.getElementById("unlearning-participants-percentage-title"),
                container: document.getElementById("unlearning-participants-percentage-container"),
            },
            unlearningRound: {
                title: document.getElementById("unlearning-round-title"),
                container: document.getElementById("unlearning-round-container"),
            },
            gradientClipVal: {
                title: document.getElementById("gradient-clip-val-title"),
                container: document.getElementById("gradient-clip-val-container"),
                help: document.getElementById("gradient-clip-val-help"),
            },
            weightFactor: {
                title: document.getElementById("weight-factor-title"),
                container: document.getElementById("weight-factor-container"),
                help: document.getElementById("weight-factor-help"),
            },
            retrainingMethod: {
                title: document.getElementById("retraining-method-title"),
                container: document.getElementById("retraining-method-container"),
            },
            retrainingRounds: {
                title: document.getElementById("retraining-rounds-title"),
                container: document.getElementById("retraining-rounds-container"),
            },
            alpha: {
                title: document.getElementById("alpha-title"),
                container: document.getElementById("alpha-container"),
                help: document.getElementById("alpha-help"),
            },
            temperature: {
                title: document.getElementById("temperature-title"),
                container: document.getElementById("temperature-container"),
                help: document.getElementById("temperature-help"),
            },
        };

        hideElements(elements);

        switch (document.getElementById("unlearning-method-select").value) {
            case UNLEARNING_METHODS.NO_UNLEARNING:
                break;

            case UNLEARNING_METHODS.PARAMETER_RESETTING:
                showCommonUnlearningElements(elements);
                break;

            case UNLEARNING_METHODS.GRADIENT_ASCENT:
                showCommonUnlearningElements(elements);
                showElements(elements, ['gradientClipVal', 'weightFactor']);
                break;
        }
    }

    function showCommonUnlearningElements(elements) {
        showElements(elements, ['unlearnAttackers', 'unlearningRound', 'retrainingMethod']);

        if (!document.getElementById("unlearn-attackers-container").checked) {
            showElements(elements, ['unlearningParticipants']);
        }

        switch (document.getElementById("retraining-method-container").value) {
            case RETRAINING_METHODS.NO_RETRAINING:
                break;

            case RETRAINING_METHODS.KNOWLEDGE_DISTILLATION:
                showElements(elements, ['retrainingRounds', 'alpha', 'temperature']);
                break;
        }
    }

    function hideElements(elements) {
        Object.values(elements).forEach(element => {
            element.title.style.display = "none";
            element.container.style.display = "none";
            if (element.help) {
                element.help.style.display = "none";
            }
        });
    }

    function showElements(elements, elementKeys) {
        elementKeys.forEach(key => {
            elements[key].title.style.display = "block";
            elements[key].container.style.display = "block";
            if (elements[key].help) {
                elements[key].help.style.display = "block";
            }
        });
    }

    function initializeEventListeners() {
        document.getElementById("unlearning-method-select").addEventListener("change", function() {
            updateUnlearnUI();
        });

        document.getElementById("unlearn-attackers-container").addEventListener("change", function() {
            updateUnlearnUI();
        });

        document.getElementById("retraining-method-container").addEventListener("change", function() {
            updateUnlearnUI();
        });
    }

    function getUnlearningConfig() {
        const unlearningMethod = document.getElementById("unlearning-method-select").value;
        const config = {
            unlearning_method: unlearningMethod,
        };

        switch(unlearningMethod) {
            case UNLEARNING_METHODS.NO_UNLEARNING:
                break;

            case UNLEARNING_METHODS.PARAMETER_RESETTING:
                getCommonUnlearningParams(config);
                break;

            case UNLEARNING_METHODS.GRADIENT_ASCENT:
                getCommonUnlearningParams(config);
                config.gradient_clip_val = parseFloat(document.getElementById("gradient-clip-val-container").value);
                config.weight_factor = parseInt(document.getElementById("weight-factor-container").value);
                break;
        }

        return config;
    }

    function getCommonUnlearningParams(config) {
        const unlearnAttackers = document.getElementById("unlearn-attackers-container").checked;
        config.unlearn_attackers = unlearnAttackers;
        if (!unlearnAttackers) {
            config.unlearning_participants_percentage = parseInt(document.getElementById("unlearning-participants-percentage-container").value);
        }
        config.unlearning_round = parseInt(document.getElementById("unlearning-round-container").value);

        const retrainingMethod = document.getElementById("retraining-method-container").value;
        config.retraining_method = retrainingMethod;
        switch (retrainingMethod) {
            case RETRAINING_METHODS.NO_RETRAINING:
                break;

            case RETRAINING_METHODS.KNOWLEDGE_DISTILLATION:
                config.retraining_rounds = parseInt(document.getElementById("retraining-rounds-container").value);
                config.alpha = parseFloat(document.getElementById("alpha-container").value);
                config.temperature = parseFloat(document.getElementById("temperature-container").value);
                break;
        }
    }

    function setUnlearningConfig(config) {
        if (!config) return;

        document.getElementById("unlearning-method-select").value = config.unlearning_method;

        switch(config.unlearning_method) {
            case UNLEARNING_METHODS.NO_UNLEARNING:
                break;

            case UNLEARNING_METHODS.PARAMETER_RESETTING:
                setCommonUnlearningParams(config);
                break;

            case UNLEARNING_METHODS.GRADIENT_ASCENT:
                setCommonUnlearningParams(config);
                document.getElementById("gradient-clip-val-container").value = config.weight_factor || 1;
                document.getElementById("weight-factor-container").value = config.weight_factor || 10;
                break;
        }

        updateUnlearnUI();
    }

    function setCommonUnlearningParams(config) {
        document.getElementById("unlearn-attackers-container").checked = config.unlearn_attackers || false;

        if (!document.getElementById("unlearn-attackers-container").checked) {
            document.getElementById("unlearning-participants-percentage-container").value = config.unlearning_participants_percentage || 10;
        }
        document.getElementById("unlearning-round-container").value = config.unlearning_round || 5;

        document.getElementById("retraining-method-container").value = config.retraining_method || RETRAINING_METHODS.NO_RETRAINING;
        switch (document.getElementById("retraining-method-container").value) {
            case RETRAINING_METHODS.NO_RETRAINING:
                break;

            case RETRAINING_METHODS.KNOWLEDGE_DISTILLATION:
                document.getElementById("retraining-rounds-container").value = config.retraining_rounds || 1;
                document.getElementById("alpha-container").value = config.alpha || 0;
                document.getElementById("temperature-container").value = config.temperature || 4;
                break;
        }
    }

    function resetUnlearningConfig() {
        document.getElementById("unlearning-method-select").value = UNLEARNING_METHODS.NO_UNLEARNING;
        updateUnlearnUI();
    }

    return {
        UNLEARNING_METHODS,
        RETRAINING_METHODS,
        initializeEventListeners,
        updateUnlearnUI,
        getUnlearningConfig,
        setUnlearningConfig,
        resetUnlearningConfig
    };
})();

export default UnlearningManager;