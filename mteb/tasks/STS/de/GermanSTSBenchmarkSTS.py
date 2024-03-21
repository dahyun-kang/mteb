from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class GermanSTSBenchmarkSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="GermanSTSBenchmark",
        hf_hub_name="jinaai/german-STSbenchmark",
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into German. "
        "Translations were originally done by T-Systems on site services GmbH.",
        reference="https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["de"],
        main_score="cosine_spearman",
        revision="e36907544d44c3a247898ed81540310442329e20",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
